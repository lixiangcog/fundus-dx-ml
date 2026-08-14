import { useCallback, useRef, useState } from 'react';

function timestamp() {
  return new Date().toLocaleTimeString('zh-CN', {
    hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit',
  });
}

export function useModuleLog(scope, maxEntries = 80) {
  const sequence = useRef(1);
  const makeEntry = useCallback((level, message, detail = '') => ({
    id: `${Date.now()}-${sequence.current++}`,
    time: timestamp(), level, message, detail,
  }), []);
  const [entries, setEntries] = useState(() => [{
    id: `initial-${Date.now()}`, time: timestamp(), level: 'info',
    message: `${scope}日志通道已建立`, detail: '等待操作',
  }]);
  const write = useCallback((level, message, detail = '') => {
    const entry = makeEntry(level, message, detail);
    setEntries((current) => [...current, entry].slice(-maxEntries));
  }, [makeEntry, maxEntries]);
  const clear = useCallback(() => {
    setEntries([makeEntry('info', `${scope}日志已清空`, '继续监听运行状态')]);
  }, [makeEntry, scope]);
  return { entries, write, clear };
}
