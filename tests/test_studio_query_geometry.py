import copy
from camera_designer.studio_query_geometry import query_geometry_identity

def fixture():
 return [dict(owner=2,v0=[0,0,0],e1=[1,0,0],e2=[0,1,0]),dict(owner=7,v0=[0,0,1],e1=[1,0,0],e2=[0,1,0])],[[0]*12],[2,7]

def test_query_identity_ignores_descriptor_order_but_binds_owner_geometry():
 t,n,i=fixture();a=query_geometry_identity(t,n,i)
 assert a==query_geometry_identity(t[::-1],n,i)
 changed=copy.deepcopy(t);changed[0]['owner']=3
 assert a['query_geometry_hash']!=query_geometry_identity(changed,n,i)['query_geometry_hash']
 changed=copy.deepcopy(t);changed[0]['v0'][0]=.001
 assert a['query_geometry_hash']!=query_geometry_identity(changed,n,i)['query_geometry_hash']

def test_component_identity_is_independent_and_binds_traversal_order():
 t,n,i=fixture();a=query_geometry_identity(t,n,i);b=query_geometry_identity(t,n,i[::-1])
 assert a['query_geometry_parts']['triangles']==b['query_geometry_parts']['triangles']
 assert a['query_geometry_parts']['bvh_nodes']==b['query_geometry_parts']['bvh_nodes']
 assert a['query_geometry_hash']!=b['query_geometry_hash']
 assert a['query_geometry_hash']!=query_geometry_identity(t,n,i,1e-6)['query_geometry_hash']
