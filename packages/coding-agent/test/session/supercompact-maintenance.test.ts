import { afterAll, beforeAll, beforeEach, describe, expect, it } from "bun:test";
import * as path from "node:path";
import { Agent, type AgentMessage } from "@oh-my-pi/pi-agent-core";
import type { AssistantMessage, Model, ToolResultMessage, UserMessage } from "@oh-my-pi/pi-ai";
import { getBundledModel } from "@oh-my-pi/pi-catalog/models";
import { ModelRegistry } from "@oh-my-pi/pi-coding-agent/config/model-registry";
import { Settings } from "@oh-my-pi/pi-coding-agent/config/settings";
import { AuthStorage } from "@oh-my-pi/pi-coding-agent/session/auth-storage";
import { SessionMaintenance, type SessionMaintenanceHost } from "@oh-my-pi/pi-coding-agent/session/session-maintenance";
import { SessionManager } from "@oh-my-pi/pi-coding-agent/session/session-manager";
import { TempDir } from "@oh-my-pi/pi-utils";

const CONTEXT_WINDOW = 100_000;

function userMessage(text: string): UserMessage {
	return { role: "user", content: [{ type: "text", text }], timestamp: Date.now() };
}

function billedAssistant(model: Model): AssistantMessage {
	return {
		role: "assistant",
		content: [{ type: "text", text: "final answer" }],
		timestamp: Date.now(),
		model: model.id,
		provider: model.provider,
		api: model.api,
		usage: {
			input: 40_000,
			output: 200,
			cacheRead: 0,
			cacheWrite: 0,
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
		},
		stopReason: "stop",
	} as AssistantMessage;
}

function assistantWithCall(id: string, model: Model): AssistantMessage {
	return {
		role: "assistant",
		content: [
			{ type: "text", text: "calling a tool" },
			{ type: "thinking", thinking: "reasoning that should not survive" },
			{ type: "toolCall", id, name: "read", arguments: { path: "src/main.ts" } },
		],
		timestamp: Date.now(),
		model: model.id,
		provider: model.provider,
		api: model.api,
		usage: {
			input: 0,
			output: 0,
			cacheRead: 0,
			cacheWrite: 0,
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
		},
		stopReason: "toolUse",
	} as AssistantMessage;
}

function toolResult(id: string): ToolResultMessage {
	return {
		role: "toolResult",
		content: [{ type: "text", text: "file body ".repeat(400) }],
		toolCallId: id,
		toolName: "read",
		isError: false,
		timestamp: Date.now(),
	} as ToolResultMessage;
}

describe("automatic supercompact at the session layer", () => {
	let authStorage: AuthStorage;
	let modelRegistry: ModelRegistry;
	let model: Model;
	let temp: TempDir;
	let sessionManager: SessionManager;
	let maintenance: SessionMaintenance;
	let agent: Agent;
	let anchoredRewrites: number[];

	function build(manager: SessionManager): SessionMaintenance {
		sessionManager = manager;
		agent = new Agent({
			initialState: { model, systemPrompt: ["Test"], tools: [], messages: [] },
		});
		agent.replaceMessages(manager.buildSessionContext().messages as AgentMessage[]);
		anchoredRewrites = [];
		const settings = Settings.isolated({
			"compaction.enabled": true,
			"compaction.asyncEnabled": false,
			"compaction.methodOrder": ["supercompact"],
			"compaction.supercompactKeepRecentTurns": 0,
		});
		const host = {
			agent,
			sessionManager,
			settings,
			modelRegistry,
			extensionRunner: undefined,
			providerSessionState: new Map(),
			model: () => model,
			thinkingLevel: () => undefined,
			isDisposed: () => false,
			isStreaming: () => false,
			isGeneratingHandoff: () => false,
			promptGeneration: () => 0,
			sessionId: () => sessionManager.getSessionId(),
			messages: () => agent.state.messages,
			baseSystemPrompt: () => ["Test"],
			nonMessageTokenSource: () => ({}),
			emitSessionEvent: async () => {},
			emitNotice: () => {},
			schedulePostPromptTask: () => {},
			scheduleAgentContinue: () => {},
			scheduleCompactionContinuation: () => false,
			buildDisplaySessionContext: () => sessionManager.buildSessionContext(),
			closeCodexProviderSessionsForHistoryRewrite: () => {},
			resetCodexProviderAfterCompaction: () => {},
			syncTodoPhasesFromBranch: () => {},
			resetAdvisorRuntimes: () => {},
			recordAnchoredHistoryRewrite: (tokens: number) => {
				anchoredRewrites.push(tokens);
			},
			getContextUsage: () => undefined,
			getContextBreakdown: () => undefined,
			abort: async () => {},
			abortHandoff: () => {},
		} as unknown as SessionMaintenanceHost;
		return new SessionMaintenance(host);
	}

	function seed(manager: SessionManager): void {
		manager.appendMessage(userMessage("first question"));
		manager.appendMessage(assistantWithCall("c1", model));
		manager.appendMessage(toolResult("c1"));
		manager.appendMessage(userMessage("second question"));
		manager.appendMessage(assistantWithCall("c2", model));
		manager.appendMessage(toolResult("c2"));
		manager.appendMessage(billedAssistant(model));
	}

	beforeAll(async () => {
		authStorage = await AuthStorage.create(":memory:");
		authStorage.setRuntimeApiKey("anthropic", "test-key");
		modelRegistry = new ModelRegistry(authStorage);
		const bundled = getBundledModel("anthropic", "claude-sonnet-4-5");
		if (!bundled) throw new Error("Expected built-in model");
		model = { ...bundled, contextWindow: CONTEXT_WINDOW };
	});

	afterAll(async () => {
		authStorage.close();
		if (temp) await temp.remove();
	});

	beforeEach(() => {
		temp = TempDir.createSync("omp-supercompact-maintenance");
	});

	it("refuses to run when the session has nowhere to write the archive", async () => {
		const manager = SessionManager.inMemory(path.join(temp.path(), "project"));
		seed(manager);
		maintenance = build(manager);
		const before = manager.getBranch().length;

		expect(maintenance.supercompactContext({ archive: true })).rejects.toThrow(/artifact directory/);
		expect(manager.getBranch().length).toBe(before);
		const results = manager
			.getBranch()
			.filter(entry => entry.type === "message" && entry.message.role === "toolResult");
		expect(results.length).toBe(2);
	});

	it("archives before it mutates, and reports the archive id", async () => {
		const manager = SessionManager.create(path.join(temp.path(), "project"), path.join(temp.path(), "sessions"));
		seed(manager);
		maintenance = build(manager);

		const outcome = await maintenance.supercompactContext({ archive: true });

		expect(outcome.toolPairsRemoved).toBe(2);
		expect(outcome.reasoningBlocksRemoved).toBe(2);
		expect(outcome.artifactId).toBeDefined();
		const archived = await Bun.file(
			path.join(manager.getArtifactsDir() ?? "", `${outcome.artifactId}.supercompact.log`),
		).text();
		expect(archived).toContain("file body");
		expect(archived).toContain("reasoning that should not survive");
		expect(archived).toContain("src/main.ts");
	});

	it("leaves the session byte-for-byte unchanged when cancelled before the mutation", async () => {
		const manager = SessionManager.create(path.join(temp.path(), "project"), path.join(temp.path(), "sessions"));
		seed(manager);
		maintenance = build(manager);
		const sessionFile = manager.getSessionFile();
		if (!sessionFile) throw new Error("Expected a persisted session file");
		const before = await Bun.file(sessionFile).text();
		const messagesBefore = agent.state.messages.length;

		const controller = new AbortController();
		controller.abort();
		const outcome = await maintenance.supercompactContext({ signal: controller.signal, archive: true });

		expect(outcome.toolPairsRemoved).toBe(0);
		expect(outcome.reasoningBlocksRemoved).toBe(0);
		expect(await Bun.file(sessionFile).text()).toBe(before);
		expect(agent.state.messages.length).toBe(messagesBefore);
		expect(anchoredRewrites.length).toBe(0);
	});

	it("bills the anchored rewrite for what it actually removed", async () => {
		const manager = SessionManager.create(path.join(temp.path(), "project"), path.join(temp.path(), "sessions"));
		seed(manager);
		maintenance = build(manager);

		const outcome = await maintenance.supercompactContext({ archive: true });

		expect(anchoredRewrites.length).toBe(1);
		expect(anchoredRewrites[0]).toBeGreaterThan(0);
		expect(anchoredRewrites[0]).toBeLessThanOrEqual(outcome.tokensBefore - outcome.tokensAfter);
	});

	it("removes the result entries from the branch instead of hollowing them out", async () => {
		const manager = SessionManager.create(path.join(temp.path(), "project"), path.join(temp.path(), "sessions"));
		seed(manager);
		maintenance = build(manager);
		const before = manager.getBranch().length;

		await maintenance.supercompactContext({ archive: true });

		const branch = manager.getBranch();
		expect(branch.length).toBe(before - 2);
		expect(branch.some(entry => entry.type === "message" && entry.message.role === "toolResult")).toBe(false);
		// Every surviving entry still hangs off something that is still there, or
		// the loader cannot walk the branch back.
		const ids = new Set(branch.map(entry => entry.id));
		for (const entry of branch) {
			if (entry.parentId === null) continue;
			expect(ids.has(entry.parentId)).toBe(true);
		}
	});

	it("leaves no tool call without a result, which providers reject", async () => {
		const manager = SessionManager.create(path.join(temp.path(), "project"), path.join(temp.path(), "sessions"));
		seed(manager);
		maintenance = build(manager);

		await maintenance.supercompactContext({ archive: true });

		const messages = manager.buildSessionContext().messages;
		const callIds = new Set<string>();
		for (const message of messages) {
			if (message.role !== "assistant") continue;
			for (const block of message.content) {
				if (block.type === "toolCall") callIds.add(block.id);
			}
		}
		expect(callIds.size).toBe(0);
		expect(messages.some(message => message.role === "toolResult")).toBe(false);
	});

	it("keeps the recent rounds whole when configured to", async () => {
		const manager = SessionManager.create(path.join(temp.path(), "project"), path.join(temp.path(), "sessions"));
		seed(manager);
		maintenance = build(manager);

		const outcome = await maintenance.supercompactContext({ keepRecentTurns: 1, archive: true });

		expect(outcome.toolPairsRemoved).toBe(1);
	});
});
