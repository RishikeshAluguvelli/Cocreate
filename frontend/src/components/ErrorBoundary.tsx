import { Component, type ReactNode } from 'react'

/** Last line of defence: a render error shows a message with a reload button instead of a blank page. */
export default class ErrorBoundary extends Component<{ children: ReactNode }, { error: Error | null }> {
  state = { error: null as Error | null }
  static getDerivedStateFromError(error: Error) { return { error } }
  render() {
    if (!this.state.error) return this.props.children
    return (
      <div className="grid h-full place-items-center p-6 text-center">
        <div className="card max-w-md p-6">
          <div className="text-lg font-semibold">Something went wrong in the interface</div>
          <p className="mt-2 text-sm text-zinc-400">Your shoots and edits are saved on the server; nothing is lost. Reload to continue.</p>
          <pre className="mt-3 max-h-32 overflow-auto rounded bg-zinc-950 p-2 text-left text-[11px] text-zinc-500">{String(this.state.error?.message ?? this.state.error)}</pre>
          <button className="btn btn-primary mt-4" onClick={() => { location.hash = ''; location.reload() }}>Reload</button>
        </div>
      </div>
    )
  }
}
