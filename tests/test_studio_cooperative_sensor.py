import pytest
from camera_designer.studio_cooperative_sensor import client_time_ms,MailboxHTTPError

@pytest.mark.parametrize('value',[None,[],7,True,{'client_compute_ms':True},{'client_compute_ms':10**1000},{'client_compute_ms':float('nan')},{'client_compute_ms':float('inf')},{'client_compute_ms':-1},{'client_compute_ms':'2'}])
def test_optional_untrusted_telemetry_cannot_abort_or_authorize(value):
    assert client_time_ms(value) is None

def test_bounded_advisory_duration():
    assert client_time_ms({'client_compute_ms':0})==0
    assert client_time_ms({'client_compute_ms':123.5})==123.5

def test_triangle_validation_window_scales_with_geometry():
    import numpy as np
    from camera_designer.studio_completed_hit_verifier import triangle_rounding_bound
    def bound(scale):
        records={0:{'v0':[0,0,scale],'e1':[scale,0,0],'e2':[0,scale,0]}}
        rays=np.array([[0,0,0,0,0,0,1,100]],np.float32)
        return triangle_rounding_bound(rays,records,np.array([0]),np.array([scale],np.float32))[0]
    large,small=bound(1),bound(.001)
    assert 0<large<1e-4
    assert small/large==pytest.approx(.001,rel=.05)

def test_triangle_validation_window_fails_closed_for_unknown_or_parallel():
    import numpy as np
    from camera_designer.studio_completed_hit_verifier import triangle_rounding_bound
    records={0:{'v0':[0,0,1],'e1':[1,0,0],'e2':[0,1,0]}}
    rays=np.array([[0,0,0,0,1,0,0,100]],np.float32)
    assert np.isnan(triangle_rounding_bound(rays,records,np.array([0]),np.array([1],np.float32))[0])
    assert np.isnan(triangle_rounding_bound(rays,records,np.array([9]),np.array([1],np.float32))[0])

def test_authorized_participant_acceptance_does_not_retrace():
    import numpy as np
    from types import SimpleNamespace
    from camera_designer.studio_completed_hit_verifier import AuthorizedParticipantAcceptance
    calls=[]
    loan=SimpleNamespace(geometry=[{'owner':3}],_assert_owner=lambda:calls.append('owner'))
    accept=AuthorizedParticipantAcceptance(loan)
    rays=np.zeros((2,8),np.float32)
    assert accept(rays,np.array([[7,3,1,3],[0,-1,0,-1]],np.float32))
    assert not accept(rays,np.array([[7,4,1,4],[0,-1,0,-1]],np.float32))
    assert accept.checked_rays==0 and accept.seconds==0 and accept.structural_rays==2
    assert calls==['owner','owner']


def test_publish_retains_exact_lease_on_explicit_not_ready():
    from camera_designer.studio_cooperative_sensor import publish_retained_when_ready,MailboxHTTPError
    from types import SimpleNamespace
    now=[0.];calls=[];work={'token':'unchanged','sample_prefix':'same-strip'}
    stop=SimpleNamespace(is_set=lambda:False,wait=lambda seconds:now.__setitem__(0,now[0]+seconds))
    def mail(action,body):
        calls.append((action,body))
        if action=='ready':return {'ready':True,'state':'committed'}
        if sum(a=='publish' for a,b in calls)==1:raise MailboxHTTPError('publish',409,'Browser has not requested another batch')
        return {'published':True,'committed':False}
    assert publish_retained_when_ready(mail,work,'query',60,stop,clock=lambda:now[0])
    publications=[b for a,b in calls if a=='publish']
    assert len(publications)==2 and all(b['work'] is work for b in publications)
    assert publications[1]['seconds']<publications[0]['seconds']

@pytest.mark.parametrize('error',[RuntimeError('lost response'),MailboxHTTPError('publish',409,'Unconsumed batch already exists')])
def test_publication_never_retries_uncertain_or_different_conflict(error):
    from camera_designer.studio_cooperative_sensor import publish_retained_when_ready
    from types import SimpleNamespace
    calls=[]
    def mail(action,body):
        if action=='ready':return {'ready':True}
        calls.append(action);raise error
    with pytest.raises(RuntimeError):publish_retained_when_ready(mail,{},'query',60,SimpleNamespace(is_set=lambda:False,wait=lambda _:None),clock=lambda:0)
    assert calls==['publish']

def test_unpublished_expiry_returns_to_existing_owner_reissue():
    from camera_designer.studio_cooperative_sensor import publish_retained_when_ready
    from types import SimpleNamespace
    now=[0.];calls=[]
    def mail(action,body):calls.append(action);return {'ready':False,'state':'committed'}
    stop=SimpleNamespace(is_set=lambda:False,wait=lambda n:now.__setitem__(0,now[0]+n))
    assert not publish_retained_when_ready(mail,{},'query',.2,stop,clock=lambda:now[0])
    assert 'publish' not in calls
