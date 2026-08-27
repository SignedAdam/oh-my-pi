/**
 * Supercompaction: reduce a session to the conversation inside it.
 *
 * Every reduction in this directory cuts on time: pick a boundary, summarize or
 * discard what is older, keep the recent tail verbatim. This one cuts on
 * content kind across the whole branch, with no recency window and no size
 * gate. Tool calls and their results are deleted as pairs, reasoning blocks are
 * deleted, and every user and assistant message survives verbatim from the
 * first turn.
 *
 * Pairs, not halves: a provider rejects an assistant tool call with no matching
 * result and a result with no matching call, so the two are only ever removed
 * together. That is also why nothing is left behind in their place.
 *
 * This module is the pure layer, region detection and in-place mutation only.
 * Persistence and context replay belong to the caller
 * (`SessionMaintenance.supercompactContext`). Layering mirrors `pruning.ts`:
 * no I/O here.
 */

import type { AssistantMessage, ToolCall, ToolResultMessage } from "@oh-my-pi/pi-ai";
import type { Tokenizer } from "../tokenizer";
import type { AgentMessage } from "../types";
import type { SessionEntry, SessionMessageEntry } from "./entries";
import { invalidateMessageCache } from "./message-cache";

/**
 * `skill` results are instructions the agent is still following, not tool
 * output, so they are never removed. Nothing else is exempt: the point of the
 * operation is that size and recency do not earn an exemption.
 */
const PROTECTED_TOOLS = ["skill"];

/** One tool call and its result, removed together. */
export interface ToolPairRegion {
	kind: "toolPair";
	/** Assistant entry holding the call. */
	callEntry: SessionMessageEntry;
	/** Index of the `toolCall` block inside that assistant message. */
	blockIndex: number;
	/** Result entry, absent when the call was never answered. */
	resultEntry?: SessionMessageEntry;
	tokens: number;
	/** Tool name, for the archive heading. */
	label: string;
	/** Call and result serialized, for the archive. */
	originalText: string;
}

/** One assistant reasoning block. */
export interface ReasoningRegion {
	kind: "reasoning";
	entry: SessionMessageEntry;
	blockIndex: number;
	tokens: number;
	label: string;
	originalText: string;
}

export type SupercompactRegion = ToolPairRegion | ReasoningRegion;

export interface SupercompactTally {
	toolPairs: number;
	reasoningBlocks: number;
	/** Tokens held by every region before the pass. */
	tokensBefore: number;
}

/**
 * Index where the Nth-from-last conversation turn begins, or 0 when the branch
 * holds fewer turns than that (every round is inside the keep window, so
 * nothing is old enough to remove).
 *
 * Only a real `user` or `bashExecution` message starts a turn here. Compaction's
 * `findTurnStartIndex` also counts `custom_message` and `branch_summary`, which
 * is right for cutting a summary boundary but wrong for this window: a trailing
 * system reminder would become the boundary and the actual last round would be
 * reduced despite the setting promising otherwise.
 */
function keepWindowStart(entries: SessionEntry[], keepTurns: number): number {
	let found = 0;
	for (let index = entries.length - 1; index >= 0; index--) {
		const entry = entries[index];
		if (entry.type !== "message") continue;
		const role = entry.message.role as string;
		if (role !== "user" && role !== "bashExecution") continue;
		found++;
		if (found === keepTurns) return index;
	}
	return 0;
}

/**
 * Locate every removable region on a branch, in document order.
 *
 * There is no minimum savings and no compaction boundary to respect.
 * Pre-boundary entries are collected too: they are not on the wire today, but
 * they are the bulk of the transcript, and leaving them means a later branch
 * navigation or reset boundary resurrects everything this was asked to remove.
 *
 * The one exemption is `keepRecentTurns`, which stops the walk at the start of
 * the Nth-from-last turn so recent rounds stay whole.
 */
export function collectSupercompactRegions(
	entries: SessionEntry[],
	tokenizer: Tokenizer,
	keepRecentTurns: number,
): SupercompactRegion[] {
	const regions: SupercompactRegion[] = [];
	const stopIndex = keepRecentTurns > 0 ? keepWindowStart(entries, keepRecentTurns) : entries.length;

	// Results are found by call id so a pair can be removed together even when
	// other entries sit between them.
	const resultByCallId = new Map<string, SessionMessageEntry>();
	for (let index = 0; index < stopIndex; index++) {
		const entry = entries[index];
		if (entry.type !== "message") continue;
		const message = entry.message as AgentMessage;
		if (message.role !== "toolResult") continue;
		resultByCallId.set((message as ToolResultMessage).toolCallId, entry as SessionMessageEntry);
	}

	for (let index = 0; index < stopIndex; index++) {
		const entry = entries[index];
		if (entry.type !== "message") continue;
		const message = entry.message as AgentMessage;
		if (message.role !== "assistant") continue;
		const assistant = message as AssistantMessage;

		for (let blockIndex = 0; blockIndex < assistant.content.length; blockIndex++) {
			const block = assistant.content[blockIndex];

			if (block.type === "toolCall") {
				const call = block as ToolCall;
				if (PROTECTED_TOOLS.includes(call.name)) continue;
				// Computer-use calls replay from `providerMetadata.actions`, so the
				// stored content is not what the provider reads.
				if (call.providerMetadata?.type === "computer") continue;
				const resultEntry = resultByCallId.get(call.id);
				const result = resultEntry?.message as ToolResultMessage | undefined;
				if (result?.providerMetadata?.type === "computer") continue;
				regions.push({
					kind: "toolPair",
					callEntry: entry as SessionMessageEntry,
					blockIndex,
					resultEntry,
					tokens:
						tokenizer.countTokens(JSON.stringify(call)) +
						(result ? tokenizer.countMessage(result as AgentMessage) : 0),
					label: call.name,
					originalText: JSON.stringify({ call, result }, null, 1),
				});
				continue;
			}

			if (block.type === "thinking" || block.type === "redactedThinking") {
				regions.push({
					kind: "reasoning",
					entry: entry as SessionMessageEntry,
					blockIndex,
					tokens: tokenizer.countTokens(JSON.stringify(block)),
					label: block.type,
					originalText: JSON.stringify(block, null, 1),
				});
			}
		}
	}

	return regions;
}

/**
 * Delete every located region and return the counts.
 *
 * Blocks are removed by index, so all removals for one assistant message are
 * collected first and applied in a single filter. Result entries are emptied
 * rather than spliced out of the branch, because the branch is a parent-linked
 * chain and dropping a node would orphan everything after it; an entry whose
 * message has no content is skipped by the provider serializers.
 */
export function applySupercompactRegions(regions: SupercompactRegion[]): SupercompactTally {
	const tally: SupercompactTally = { toolPairs: 0, reasoningBlocks: 0, tokensBefore: 0 };
	const blocksToDrop = new Map<SessionMessageEntry, Set<number>>();
	const resultsToEmpty = new Set<SessionMessageEntry>();

	for (const region of regions) {
		tally.tokensBefore += region.tokens;
		const entry = region.kind === "toolPair" ? region.callEntry : region.entry;
		let indexes = blocksToDrop.get(entry);
		if (!indexes) {
			indexes = new Set();
			blocksToDrop.set(entry, indexes);
		}
		indexes.add(region.blockIndex);
		if (region.kind === "toolPair") {
			tally.toolPairs++;
			if (region.resultEntry) resultsToEmpty.add(region.resultEntry);
		} else {
			tally.reasoningBlocks++;
		}
	}

	for (const [entry, indexes] of blocksToDrop) {
		const message = entry.message as AssistantMessage;
		// Provider serializers omit empty assistant turns, so a message left with
		// no blocks is correct: never invent model-authored text.
		message.content = message.content.filter((_block, index) => !indexes.has(index));
		// A native-history copy would replay the original blocks and undo this.
		message.providerPayload = undefined;
		invalidateMessageCache(message as AgentMessage);
	}

	for (const entry of resultsToEmpty) {
		const message = entry.message as ToolResultMessage;
		message.content = [];
		message.prunedAt = Date.now();
		invalidateMessageCache(message as AgentMessage);
	}

	return tally;
}
