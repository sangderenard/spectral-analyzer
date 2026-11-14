import unittest
from nodus_console_station import NodusConsoleMenu, workspace_from_graph

class ConsoleTests(unittest.TestCase):
    def test_stream_and_settings(self):
        menu = NodusConsoleMenu({'module_ids':['7'], 'panel':{'name':'motor', 'knobs':[{'name':'gain','label':'Gain','default':1.0}], 'payload':{'actions':[{'key':'apply','label':'Apply'}]}}})
        calls=[]
        menu.bind_host(subscribe=lambda *args: calls.append(args))
        menu.activate('nodus.stream_stats')
        menu.update_stream('stats', {'tick': 12})
        menu.update_values({'gain': 0.5})
        self.assertEqual(menu.knob_values['gain'], 0.5)
        self.assertIn('12', menu.knob_values['feed_stats'])
        self.assertEqual(calls, [(('7',),'stats',True)])
        self.assertFalse(menu.activate('motor.apply'))
    def test_real_workspace(self):
        graph={'modules':[{'stable_id':7,'label':'Mixer','rows':[]}]}
        ws=workspace_from_graph(graph)
        obj=ws._objects['nodus/7']
        self.assertEqual(obj.station_type,'nodus_console')
        self.assertEqual(obj.to_dict()['build_state']['module_ids'],['7'])
    def test_unknown_module(self):
        with self.assertRaises(ValueError):
            workspace_from_graph({'modules':[]},consoles={'x':{'module_ids':['7']}})



class FeedTests(unittest.TestCase):
    def test_actual_protocol_methods_and_opt_in(self):
        from nodus_console_feeds import NodusConsoleFeeds
        class Client:
            def __init__(self): self.calls=[]
            def graph(self): self.calls.append('GRAPH'); return {'revision':3,'modules':[{'stable_id':7},{'stable_id':9}]}
            def tools(self): self.calls.append('TOOLS'); return {'complete':False,'tools':[]}
            def activity(self): self.calls.append('ACTIVITY'); return {'tick':42}
        client=Client(); feeds=NodusConsoleFeeds(client)
        menu=NodusConsoleMenu({'module_ids':['7']})
        feeds.attach(menu); feeds.poll(); self.assertEqual(client.calls,[])
        menu.activate('nodus.stream_metadata'); menu.activate('nodus.stream_stats'); feeds.poll()
        self.assertEqual(client.calls,['GRAPH','ACTIVITY'])
        self.assertEqual(menu._feeds['metadata']['modules'],[{'stable_id':7}])
        self.assertEqual(menu._feeds['stats']['activity']['tick'],42)

class FeedFallbackTests(unittest.TestCase):
    def test_native_index_fallback_does_not_consume_or_stop_other_feeds(self):
        from nodus_console_feeds import NodusConsoleFeeds
        class Client:
            fallback = True
            def graph(self):
                if self.fallback:
                    return {'definition_error': 'missing stable connector', 'revision': 4,
                            'modules': [], 'edges': [{'source': {'module_index': 0},
                                                     'target': {'module_index': 1}}]}
                return {'revision': 5, 'modules': [{'stable_id': 7}],
                        'connectors': [], 'edges': [{'source': {'module_id': 7},
                            'target': {'module_id': 9}, 'fifo': {'token_bytes': 32}}]}
            def activity(self): return {'tick': 42}
        client = Client()
        feeds = NodusConsoleFeeds(client)
        menu = NodusConsoleMenu({'module_ids': ['7']})
        feeds.attach(menu)
        for stream in ('metadata', 'abi', 'stats'):
            menu.activate('nodus.stream_' + stream)
        feeds.poll()
        self.assertFalse(menu._feeds['abi']['available'])
        self.assertFalse(menu._feeds['metadata']['available'])
        self.assertEqual(menu._feeds['stats']['activity']['tick'], 42)
        client.fallback = False
        feeds.poll()
        self.assertEqual(menu._feeds['metadata']['modules'], [{'stable_id': 7}])
        self.assertEqual(menu._feeds['abi']['edges'][0]['fifo'], {'token_bytes': 32})

class InteractiveTests(unittest.TestCase):
    def test_stepper_uses_existing_dispatch(self):
        import pygame
        menu=NodusConsoleMenu({'module_ids':['7'],'panel':{'name':'gain','knobs':[{'name':'gain','label':'Gain','default':1.0}]}})
        calls=[]; menu.bind_host(dispatch=lambda *args:calls.append(args)); menu.show_hud(True)
        menu._doc_knob_rects={'gain':{'rect':(0,0,100,20),'widget':'stepper','dtype':'float','low':0,'high':2,'step':0.25,'default':1}}
        self.assertTrue(menu.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN,button=1,pos=(90,10))))
        self.assertEqual(menu.knob_values['gain'],1.25)
        self.assertEqual(calls[0][0],('7',))
        self.assertEqual(calls[0][1],'setting.gain')
    def test_shared_room_choices(self):
        from controls import clicked_knob_value
        info={'widget':'segmented','choices':['a','b'],'dtype':'choice'}
        self.assertEqual(clicked_knob_value(info,'a',75,100),'b')
        self.assertEqual(clicked_knob_value(info,0,75,100,choice_index=True),1)
        self.assertEqual(clicked_knob_value({'widget':'readonly'},'fixed',75,100),'fixed')

if __name__=='__main__': unittest.main()
