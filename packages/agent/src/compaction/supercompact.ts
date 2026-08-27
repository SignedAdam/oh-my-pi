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
 * Exempt tools, each for a reason that is not about size or recency - the point
 * of the operation is that neither of those earns an exemption.
 *
 * `skill` results are instructions the agent is still following, not tool output.
 *
 * `task` results carry the subagent's usage in `details.usage`, which the session
 * folds into its cumulative token, cost, orchestration and premium-request
 * totals. Removing the entry would quietly reduce figures the account was already
 * billed for, so `/usage` and the status line would fall after a reduction.
 *
 * This one is not free. The results themselves are small - 0.3% of tool-result
 * bytes across a 60-session sample - but a `task` call carries the whole subagent
 * prompt in its arguments, and those stay too. On a task-heavy session that is
 * around four points of the total reduction. Correct accounting is worth more
 * than the four points; folding removed usage into retained totals would buy them
 * back and is the better fix when someone wants them.
 */
const PROTECTED_TOOLS = ["skill", "task"];

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
	/** Tool-result entries the caller must remove from the branch. */
	removedResultEntryIds: string[];
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
/**
 * Identity of one tool call for off-branch protection: the entry that holds it
 * plus its id. The id alone is not enough, because a later call can reuse it.
 */
export function sharedCallKey(entryId: string, callId: string): string {
	return `${entryId}\u0000${callId}`;
}

export function collectSupercompactRegions(
	entries: SessionEntry[],
	tokenizer: Tokenizer,
	keepRecentTurns: number,
	callsAnsweredOffBranch: ReadonlySet<string> = new Set(),
): SupercompactRegion[] {
	const regions: SupercompactRegion[] = [];
	const stopIndex = keepRecentTurns > 0 ? keepWindowStart(entries, keepRecentTurns) : entries.length;

	// One ordered pass, because pairing is positional. Ids get reused across
	// turns, so a call owns the first result for its id that arrives before the
	// next call reusing it - not the first one in a global queue. A call still
	// pending when its id is reused never produced a result and is dropped: it
	// must not take the later call's result and leave that call to be deleted
	// with its result still live, which a provider rejects as an unmatched pair.
	interface PendingCall {
		entry: SessionMessageEntry;
		blockIndex: number;
		call: ToolCall;
		exempt: boolean;
	}
	const pendingByCallId = new Map<string, PendingCall>();

	const pushPair = (pending: PendingCall, resultEntry: SessionMessageEntry | undefined): void => {
		const result = resultEntry?.message as ToolResultMessage | undefined;
		// Computer-use results replay from `providerMetadata.screenshot`, so the
		// stored content is not what the provider reads.
		if (pending.exempt || result?.providerMetadata?.type === "computer") return;
		regions.push({
			kind: "toolPair",
			callEntry: pending.entry,
			blockIndex: pending.blockIndex,
			resultEntry,
			tokens:
				tokenizer.countTokens(JSON.stringify(pending.call)) +
				(result ? tokenizer.countMessage(result as AgentMessage) : 0),
			label: pending.call.name,
			originalText: JSON.stringify({ call: pending.call, result }, null, 1),
		});
	};

	for (let index = 0; index < stopIndex; index++) {
		const entry = entries[index];
		if (entry.type !== "message") continue;
		const message = entry.message as AgentMessage;

		if (message.role === "toolResult") {
			const callId = (message as ToolResultMessage).toolCallId;
			const pending = pendingByCallId.get(callId);
			if (pending === undefined) continue;
			pendingByCallId.delete(callId);
			pushPair(pending, entry as SessionMessageEntry);
			continue;
		}

		if (message.role !== "assistant") continue;
		const assistant = message as AssistantMessage;

		for (let blockIndex = 0; blockIndex < assistant.content.length; blockIndex++) {
			const block = assistant.content[blockIndex];

			if (block.type === "toolCall") {
				const call = block as ToolCall;
				// A call already pending under this id closes here with no result.
				pendingByCallId.delete(call.id);
				pendingByCallId.set(call.id, {
					entry: entry as SessionMessageEntry,
					blockIndex,
					call,
					exempt:
						PROTECTED_TOOLS.includes(call.name) ||
						// Computer-use calls replay from `providerMetadata.actions`, so
						// the stored content is not what the provider reads.
						call.providerMetadata?.type === "computer" ||
						// The same call can be answered on more than one branch: the
						// `ask` re-answer flow adds a sibling result. The call entry is
						// shared by every branch, so deleting the block here would strip
						// it from the other branch too and leave that result unmatched.
						// Keyed by entry and id together, because a later ordinary call
						// may reuse the id and has no sibling result of its own.
						callsAnsweredOffBranch.has(sharedCallKey(entry.id, call.id)),
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

	// A call with no result cannot be removed: deleting the block would be fine,
	// but the pair is the unit and there is no result to take with it. Provider
	// serializers already strip a dangling call when it reaches context.

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
	const tally: SupercompactTally = { toolPairs: 0, reasoningBlocks: 0, tokensBefore: 0, removedResultEntryIds: [] };
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

	// The result is a whole entry, so it can only leave by being removed from the
	// branch. Emptying it instead would leave a `tool_result` with no `tool_use`,
	// which Anthropic rejects, and stamping `prunedAt` would render it as
	// `[Output truncated]` - a placeholder standing in for content that is not
	// truncated but gone. The caller removes these ids and persists.
	tally.removedResultEntryIds = [...resultsToEmpty].map(entry => entry.id);

	return tally;
}
