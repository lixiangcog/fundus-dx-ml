import { useEffect, useRef } from 'react';
import { Check, CircleAlert, SquareTerminal, Trash2 } from 'lucide-react';
import './module-log.css';

const LEVEL_LABELS = {
  info: 'INFO',
  run: 'RUN',
  success: 'OK',
  warning: 'WARN',
  error: 'ERROR',
};

function ModuleLog({ title, entries, onClear }) {
  const streamRef = useRef(null);
  useEffect(() => {
    const stream = streamRef.current;
    if (stream) stream.scrollTop = stream.scrollHeight;
  }, [entries]);

  return <section className="module-log" aria-label={`${title}运行日志`}>
    <header>
      <span><SquareTerminal size={14}/><b>{title}</b><em>运行日志</em></span>
      <div><i/><small>实时记录 · {entries.length} 条</small><button type="button" onClick={onClear} aria-label="清空运行日志"><Trash2 size={12}/>清空</button></div>
    </header>
    <div className="module-log-stream" ref={streamRef} aria-live="polite">
      {entries.map((entry) => <div className={`module-log-line ${entry.level}`} key={entry.id}>
        <time>{entry.time}</time>
        <span>{entry.level === 'success' ? <Check size={10}/> : entry.level === 'error' || entry.level === 'warning' ? <CircleAlert size={10}/> : null}{LEVEL_LABELS[entry.level] || 'INFO'}</span>
        <b>{entry.message}</b>
        {entry.detail && <small>{entry.detail}</small>}
      </div>)}
    </div>
  </section>;
}

export default ModuleLog;
