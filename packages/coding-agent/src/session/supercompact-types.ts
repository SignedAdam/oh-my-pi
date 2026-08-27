/**
 * Public shape of the `supercompact` operation, kept in a dependency-free leaf
 * module so slash-command registries and controllers can import
 * `formatSupercompactSummary` without pulling in the heavy `agent-session`
 * module graph (which would form an import cycle through the registry).
 */

/** Per-run overrides, each defaulting to its setting. */
export interface SupercompactOptions {
	/** Rewrite the current session instead of copying it first. */
	inPlace?: boolean;
	/** Leave the last N rounds untouched; overrides `compaction.supercompactKeepRecentTurns`. */
	keepRecentTurns?: number;
}

/** Context reduction achieved by one supercompact pass. */
export interface SupercompactOutcome {
	toolPairsRemoved: number;
	reasoningBlocksRemoved: number;
	tokensBefore: number;
	tokensAfter: number;
	/**
	 * Archive of everything removed. Written only for an in-place run, where the
	 * rewritten session is the only copy. A copied run needs none: the original
	 * session file is the archive.
	 */
	artifactId?: string;
}

/** Outcome of an `AgentSession.supercompact` run. */
export interface SupercompactResult extends SupercompactOutcome {
	/** True when it ran on a copy, leaving the original session untouched. */
	copied: boolean;
	/** Session file the reduced history lives in. */
	sessionFile?: string;
}

/** One-line operator summary of a {@link SupercompactResult} (shared by TUI + ACP). */
export function formatSupercompactSummary(result: SupercompactResult): string {
	const parts: string[] = [];
	if (result.toolPairsRemoved > 0) {
		parts.push(`${result.toolPairsRemoved} tool call${result.toolPairsRemoved === 1 ? "" : "s"}`);
	}
	if (result.reasoningBlocksRemoved > 0) {
		parts.push(`${result.reasoningBlocksRemoved} reasoning block${result.reasoningBlocksRemoved === 1 ? "" : "s"}`);
	}
	if (parts.length === 0) return "Nothing to remove: this session is already only conversation.";

	const saved = Math.max(0, result.tokensBefore - result.tokensAfter);
	const percent = result.tokensBefore > 0 ? Math.round((saved / result.tokensBefore) * 100) : 0;
	const where = result.copied ? "Copied the session and removed" : "Removed";
	const recovery = result.artifactId ? ` Originals: artifact://${result.artifactId}.` : "";
	return (
		`${where} ${parts.join(" and ")}. ` +
		`${result.tokensBefore.toLocaleString()} -> ${result.tokensAfter.toLocaleString()} tokens, ${percent}% smaller.` +
		recovery
	);
}
