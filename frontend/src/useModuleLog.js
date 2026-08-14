import { useCallback, useRef, useState } from 'react';

function timestamp() {
  return new Date().toLocaleTimeString('zh-CN', {
    hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit',
  });
}

export function useModuleLog(scope, maxEntries = 80) {
  const sequence = useRef(1);
  const makeEntry = useCallback((level, message, detail = '', channel = '') => ({
    id: `${Date.now()}-${sequence.current++}`,
    time: timestamp(), level, message, detail, channel,
  }), []);
  const [entries, setEntries] = useState(() => [{
    id: `initial-${Date.now()}`, time: timestamp(), level: 'info',
    channel: 'SYSTEM', message: 'terminal session attached', detail: `${scope} / idle`,
  }]);
  const write = useCallback((level, message, detail = '', channel = '') => {
    const entry = makeEntry(level, message, detail, channel);
    setEntries((current) => [...current, entry].slice(-maxEntries));
  }, [makeEntry, maxEntries]);
  const writeMany = useCallback((rows) => {
    const next = rows.map((row) => makeEntry(
      row.level || 'info', row.message, row.detail || '', row.channel || '',
    ));
    setEntries((current) => [...current, ...next].slice(-maxEntries));
  }, [makeEntry, maxEntries]);
  const clear = useCallback(() => {
    setEntries([makeEntry('info', 'clear', `${scope} / listening`, 'SYSTEM')]);
  }, [makeEntry, scope]);
  return { entries, write, writeMany, clear };
}
