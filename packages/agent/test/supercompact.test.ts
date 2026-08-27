import { describe, expect, it } from "bun:test";
import { Tokenizer } from "@oh-my-pi/pi-agent-core";
import type { SessionMessageEntry } from "@oh-my-pi/pi-agent-core/compaction/entries";
import { applySupercompactRegions, collectSupercompactRegions } from "@oh-my-pi/pi-agent-core/compaction/supercompact";
import type { AssistantMessage, TextContent, ToolResultMessage, Usage, UserMessage } from "@oh-my-pi/pi-ai";

const tokenizer = new Tokenizer();

function usage(): Usage {
	return {
		input: 0,
		output: 0,
		cacheRead: 0,
		cacheWrite: 0,
		totalTokens: 0,
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
	};
}

function messageEntry(id: string, message: AssistantMessage | ToolResultMessage | UserMessage): SessionMessageEntry {
	return { type: "message", id, parentId: null, timestamp: "2026-08-26T00:00:00.000Z", message };
}

function userTurn(id: string, text: string): SessionMessageEntry {
	return messageEntry(id, { role: "user", content: text, timestamp: 0 });
}

function assistantTurn(id: string, content: AssistantMessage["content"]): SessionMessageEntry {
	return messageEntry(id, {
		role: "assistant",
		content,
		api: "mock",
		provider: "mock",
		model: "mock-model",
		usage: usage(),
		stopReason: "toolUse",
		timestamp: 0,
	});
}

function toolOutcomeWithId(entryId: string, toolCallId: string, toolName: string, text: string): SessionMessageEntry {
	const entry = toolOutcome(toolCallId, toolName, text);
	return { ...entry, id: entryId };
}

function toolOutcome(toolCallId: string, toolName: string, text: string): SessionMessageEntry {
	const content: TextContent[] = [{ type: "text", text }];
	return messageEntry(`result-${toolCallId}`, {
		role: "toolResult",
		toolCallId,
		toolName,
		content,
		isError: false,
		timestamp: 0,
	});
}

function supercompact(entries: SessionMessageEntry[], keepRecentTurns = 0) {
	const regions = collectSupercompactRegions(entries, tokenizer, keepRecentTurns);
	return { regions, tally: applySupercompactRegions(regions) };
}

describe("supercompact", () => {
	it("deletes tool calls with their results and keeps the conversation verbatim", () => {
		const question = "Why does the build fail on arm64 but pass on x86?";
		const answer = "The native addon is only prebuilt for x86.";
		const entries = [
			userTurn("u1", question),
			assistantTurn("a1", [
				{ type: "thinking", thinking: "private reasoning ".repeat(200) },
				{ type: "text", text: answer },
				{
					type: "toolCall",
					id: "c1",
					name: "write",
					arguments: { path: "src/answer.ts", content: "x".repeat(9000) },
				},
			]),
			toolOutcome("c1", "write", "Wrote 400 lines"),
		];

		const { tally } = supercompact(entries);

		expect(tally.toolPairs).toBe(1);
		expect(tally.reasoningBlocks).toBe(1);

		// Both sides of the conversation survive byte for byte.
		expect((entries[0].message as UserMessage).content).toBe(question);
		const assistant = entries[1].message as AssistantMessage;
		expect(assistant.content).toHaveLength(1);
		expect((assistant.content[0] as TextContent).text).toBe(answer);

		// The call is gone, not hollowed out, and its result entry is named for removal.
		expect(assistant.content.some(block => block.type === "toolCall")).toBe(false);
		expect(tally.removedResultEntryIds).toEqual([entries[2].id]);
	});

	it("never leaves a call without its result or a result without its call", () => {
		const entries = [
			assistantTurn("a1", [
				{ type: "toolCall", id: "c1", name: "read", arguments: { path: "a.ts" } },
				{ type: "toolCall", id: "c2", name: "read", arguments: { path: "b.ts" } },
			]),
			toolOutcome("c1", "read", "contents of a"),
			toolOutcome("c2", "read", "contents of b"),
		];

		const { tally } = supercompact(entries);

		const calls = (entries[0].message as AssistantMessage).content.filter(block => block.type === "toolCall");
		expect(calls).toHaveLength(0);
		expect(tally.removedResultEntryIds).toEqual([entries[1].id, entries[2].id]);
	});

	it("archives the call and its result together so an in-place run can recover them", () => {
		const secret = "supersecret-payload";
		const entries = [
			assistantTurn("a1", [{ type: "toolCall", id: "c1", name: "bash", arguments: { command: secret } }]),
			toolOutcome("c1", "bash", "command output here"),
		];

		const { regions } = supercompact(entries);

		expect(regions).toHaveLength(1);
		expect(regions[0].originalText).toContain(secret);
		expect(regions[0].originalText).toContain("command output here");
	});

	it("drops the native replay payload so originals cannot come back", () => {
		const entries = [
			assistantTurn("a1", [{ type: "toolCall", id: "c1", name: "bash", arguments: { command: "ls" } }]),
		];
		(entries[0].message as AssistantMessage).providerPayload = {
			type: "openaiResponsesHistory",
			items: [{ type: "function_call", arguments: "ls" }],
		};

		supercompact(entries);

		expect((entries[0].message as AssistantMessage).providerPayload).toBeUndefined();
	});

	it("never removes skill calls, which are live instructions", () => {
		const entries = [
			assistantTurn("a1", [{ type: "toolCall", id: "c1", name: "skill", arguments: { name: "linear" } }]),
			toolOutcome("c1", "skill", "Follow these steps"),
		];

		const { tally } = supercompact(entries);

		expect(tally.toolPairs).toBe(0);
		expect((entries[1].message as ToolResultMessage).content).toHaveLength(1);
	});

	it("leaves the last N rounds whole when keepRecentTurns is set", () => {
		const round = (n: number) => [
			userTurn(`u${n}`, `round ${n}`),
			assistantTurn(`a${n}`, [
				{ type: "thinking", thinking: `reasoning ${n}` },
				{ type: "toolCall", id: `c${n}`, name: "read", arguments: { path: `f${n}.ts` } },
			]),
			toolOutcome(`c${n}`, "read", `contents ${n}`),
		];
		const entries = [...round(1), ...round(2), ...round(3)];

		const { tally } = supercompact(entries, 1);

		expect(tally.toolPairs).toBe(2);
		expect(tally.reasoningBlocks).toBe(2);

		const lastAssistant = entries[7].message as AssistantMessage;
		expect(lastAssistant.content.some(block => block.type === "toolCall")).toBe(true);
		expect(lastAssistant.content.some(block => block.type === "thinking")).toBe(true);
		expect((entries[8].message as ToolResultMessage).content).toHaveLength(1);
	});

	it("does not treat a trailing reminder as the start of the kept round", () => {
		const entries = [
			userTurn("u1", "old round"),
			assistantTurn("a1", [{ type: "toolCall", id: "c1", name: "read", arguments: { path: "old.ts" } }]),
			toolOutcome("c1", "read", "old contents"),
			userTurn("u2", "the round I want kept"),
			assistantTurn("a2", [{ type: "toolCall", id: "c2", name: "read", arguments: { path: "new.ts" } }]),
			toolOutcome("c2", "read", "new contents"),
			// A reminder is injected as a custom_message, not a real turn. Counting it
			// as a turn start would push the window past the round the setting kept.
			{
				type: "custom_message",
				id: "r1",
				parentId: null,
				timestamp: "2026-08-26T00:00:00.000Z",
				customType: "system-reminder",
				content: "be brief",
			} as unknown as SessionMessageEntry,
		];

		const { tally } = supercompact(entries, 1);

		expect(tally.toolPairs).toBe(1);
		const keptAssistant = entries[4].message as AssistantMessage;
		expect(keptAssistant.content.some(block => block.type === "toolCall")).toBe(true);
	});

	it("finds nothing to remove in a session that is already only conversation", () => {
		const entries = [userTurn("u1", "what is the plan?"), assistantTurn("a1", [{ type: "text", text: "Ship it." }])];

		expect(collectSupercompactRegions(entries, tokenizer, 0)).toHaveLength(0);
	});

	it("pairs a reused tool-call id with its own result, never stranding the earlier one", () => {
		const entries = [
			assistantTurn("a1", [{ type: "toolCall", id: "dup", name: "read", arguments: { path: "first.ts" } }]),
			toolOutcomeWithId("r1", "dup", "read", "first body"),
			assistantTurn("a2", [{ type: "toolCall", id: "dup", name: "read", arguments: { path: "second.ts" } }]),
			toolOutcomeWithId("r2", "dup", "read", "second body"),
		];

		const regions = collectSupercompactRegions(entries, tokenizer, 0);
		const tally = applySupercompactRegions(regions);

		// Both results leave. A map keyed by id would name "r2" twice and leave
		// "r1" behind as a result with no call, which a provider rejects.
		expect(tally.toolPairs).toBe(2);
		expect([...tally.removedResultEntryIds].sort()).toEqual(["r1", "r2"]);
		for (const index of [0, 2]) {
			const assistant = entries[index].message as AssistantMessage;
			expect(assistant.content.some(block => block.type === "toolCall")).toBe(false);
		}
	});

	it("archives each reused id against the arguments it was actually called with", () => {
		const entries = [
			assistantTurn("a1", [{ type: "toolCall", id: "dup", name: "read", arguments: { path: "first.ts" } }]),
			toolOutcomeWithId("r1", "dup", "read", "first body"),
			assistantTurn("a2", [{ type: "toolCall", id: "dup", name: "read", arguments: { path: "second.ts" } }]),
			toolOutcomeWithId("r2", "dup", "read", "second body"),
		];

		const regions = collectSupercompactRegions(entries, tokenizer, 0);

		expect(regions[0].originalText).toContain("first.ts");
		expect(regions[0].originalText).toContain("first body");
		expect(regions[1].originalText).toContain("second.ts");
		expect(regions[1].originalText).toContain("second body");
	});

	it("takes an exempt call's own result out of the queue", () => {
		const entries = [
			assistantTurn("a1", [{ type: "toolCall", id: "dup", name: "skill", arguments: { path: "s" } }]),
			toolOutcomeWithId("r-skill", "dup", "skill", "skill instructions"),
			assistantTurn("a2", [{ type: "toolCall", id: "dup", name: "read", arguments: { path: "x.ts" } }]),
			toolOutcomeWithId("r-read", "dup", "read", "file body"),
		];

		const regions = collectSupercompactRegions(entries, tokenizer, 0);
		const tally = applySupercompactRegions(regions);

		// The skill pair stays whole, and the read call removes its own result -
		// not the skill's, which an id-keyed queue that skipped exempt calls would.
		expect(tally.toolPairs).toBe(1);
		expect(tally.removedResultEntryIds).toEqual(["r-read"]);
		const skillCall = entries[0].message as AssistantMessage;
		expect(skillCall.content.some(block => block.type === "toolCall")).toBe(true);
		const readCall = entries[2].message as AssistantMessage;
		expect(readCall.content.some(block => block.type === "toolCall")).toBe(false);
	});
});
