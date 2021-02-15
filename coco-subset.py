import json

json_in='instances_test.json'
json_out=json_in.replace('.json','_sub.json')

with open(json_in) as f:
    data=json.load(f)

new_data={}

new_data['info']=data['info']
new_data['licences']=data['licences']
new_data['categories']=data['categories']

images_subset=data['images'][:50]
images_subset_id=[]
for i in images_subset:
    images_subset_id.append(i['id'])

annotations_subset=[]
for i in range(len(data['annotations'])):
    if data['annotations'][i]['image_id'] in images_subset_id:
        annotations_subset.append(data['annotations'][i])

new_data['images']=images_subset
new_data['annotations']=annotations_subset

with open(json_out,'w') as f:
    json.dump(new_data,f,indent=4,ensure_ascii=False)
