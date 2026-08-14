import { useEffect, useRef } from 'react';
import { SquareTerminal, Trash2 } from 'lucide-react';
import './module-log.css';

const LEVEL_LABELS = {
  command: 'SHELL', info: 'INFO', run: 'RUN', success: 'DONE', warning: 'WARN', error: 'ERROR',
};

function ModuleLog({ title, entries, onClear, running = false }) {
  const streamRef = useRef(null);
  useEffect(() => {
    const stream = streamRef.current;
    if (stream) stream.scrollTop = stream.scrollHeight;
  }, [entries]);

  return <section className={`module-log ${running ? 'is-running' : ''}`} aria-label={`${title}终端推理日志`}>
    <header>
      <span className="terminal-identity"><i/><i/><i/><SquareTerminal size={14}/><b>inference@retina-gpu</b><em>: ~/{title}</em></span>
      <div><span className="terminal-live"><i/>{running ? 'RUNNING' : 'LIVE'}</span><small>{entries.length} lines</small><button type="button" onClick={onClear} aria-label="清空终端日志"><Trash2 size={12}/>clear</button></div>
    </header>
    <div className="module-log-stream" ref={streamRef} aria-live="polite">
      {entries.map((entry) => <div className={`module-log-line ${entry.level}`} key={entry.id}>
        <time>{entry.time}</time>
        <span className="terminal-channel">[{entry.channel || LEVEL_LABELS[entry.level] || 'INFO'}]</span>
        <span className="terminal-prompt">{entry.level === 'command' ? '$' : '›'}</span>
        <b>{entry.message}</b>
        {entry.detail && <small><i>::</i>{entry.detail}</small>}
      </div>)}
      <div className="terminal-cursor" aria-hidden="true"><time>{entries[entries.length - 1]?.time || '--:--:--'}</time><span>[_]</span><i/></div>
    </div>
  </section>;
}

export default ModuleLog;
