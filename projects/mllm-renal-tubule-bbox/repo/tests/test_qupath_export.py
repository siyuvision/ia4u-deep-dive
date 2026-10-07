from conftest import load_script

def test_center_boxes_rescale_to_input_pixels_and_close_polygon():
    mod=load_script('export_qupath')
    data={'image':{'width':200,'height':100},'predictions':[{'x':100,'y':50,'width':40,'height':20}]}
    result=mod.feature_collection(data,(1000,400))
    feature=result['features'][0]
    assert feature['geometry']['coordinates']==[[[400,160],[600,160],[600,240],[400,240],[400,160]]]
    assert feature['properties']['classification']['name']=='renal tubule bbox'

def test_empty_prediction_stays_empty():
    mod=load_script('export_qupath')
    assert mod.feature_collection({'image':{'width':200,'height':100},'predictions':[]},(1000,400))['features']==[]
