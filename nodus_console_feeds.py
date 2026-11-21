"""Opt-in Pluck feeds from the existing authenticated Nodus console client.

This uses its READ-ONLY verbs; it cannot authorize parameter writes. A caller
owns/authenticates the ConsoleClient and passes exclusive ownership while this
session runs. No password/token lookup, new server, or transport is created.
"""
from threading import Event, Lock, Thread


class NodusConsoleFeeds:
    def __init__(self, client, *, interval=1.0):
        if interval <= 0: raise ValueError('interval must be positive')
        self.client, self.interval = client, float(interval)
        self._lock, self._stop = Lock(), Event()
        self._menus, self._requested = {}, {}
        self._thread = None

    def attach(self, menu, *, dispatch=None):
        with self._lock:
            self._menus[id(menu)] = menu
            self._requested[id(menu)] = set()
        def subscribe(module_ids, stream, enabled):
            with self._lock:
                if enabled: self._requested[id(menu)].add(stream)
                else: self._requested[id(menu)].discard(stream)
        menu.bind_host(dispatch=dispatch, subscribe=subscribe)

    def poll(self):
        with self._lock:
            subscriptions = [(self._menus[key], set(value)) for key,value in self._requested.items() if value]
        wanted = set().union(*(kinds for _,kinds in subscriptions)) if subscriptions else set()
        if not wanted: return
        # Fetch each existing verb at most once per sample. ACTIVITY and TOOLS
        # have no revision field; retain that fact rather than invent a join.
        graph = self.client.graph() if wanted & {'metadata','abi'} else None
        tools = self.client.tools() if 'api' in wanted else None
        activity = self.client.activity() if 'stats' in wanted else None
        for menu, kinds in subscriptions:
            ids = set(menu.module_ids)
            # A fallback GRAPH uses native indices, not stable connector IDs.
            # Do not guess an identity mapping or leave stale ABI data visible.
            unavailable = graph is not None and 'definition_error' in graph
            unavailable_streams = kinds & {'metadata', 'abi'} if unavailable else set()
            for stream in unavailable_streams:
                menu.update_stream(stream, {'available': False,
                    'revision': graph.get('revision'),
                    'reason': 'Stable graph definition unavailable; native-index fallback is not a console binding.'})
            if 'metadata' in kinds and not unavailable:
                menu.update_stream('metadata', {'revision': graph.get('revision'),
                    'modules':[m for m in graph['modules'] if str(m['stable_id']) in ids]})
            if 'abi' in kinds and not unavailable:
                menu.update_stream('abi', {'revision':graph.get('revision'),
                    'connectors':[c for c in graph.get('connectors',[]) if str(c['reference']['module_id']) in ids],
                    'edges':[e for e in graph.get('edges',[]) if str(e['source']['module_id']) in ids or str(e['target']['module_id']) in ids]})
            if 'api' in kinds: menu.update_stream('api', {'scope':'canvas','tools':tools})
            if 'stats' in kinds: menu.update_stream('stats', {'scope':'canvas','activity':activity})
            if 'profile' in kinds: menu.update_stream('profile', {'available':False,'reason':'The Nodus console has no profile verb.'})

    def start(self):
        if self._thread is not None and self._thread.is_alive(): return
        self._stop.clear()
        self._thread = Thread(target=self._run, name='nodus-console-feeds', daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            try: self.poll()
            except Exception as exc:
                # Report the category without exposing transport exception text,
                # which may contain an address or account-sensitive details.
                with self._lock: menus = list(self._menus.values())
                for menu in menus: menu.status = 'Feed unavailable: ' + type(exc).__name__
                self._stop.set()
                return
            self._stop.wait(self.interval)

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, float(getattr(self.client,'timeout',1.0))+1.0))
            if self._thread.is_alive(): raise RuntimeError('Console feed read has not returned; client ownership retained')
        self.client.close()
