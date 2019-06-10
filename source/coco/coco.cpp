#include "coco.h"
#include <rapidjson/istreamwrapper.h>
#include <fstream>
#include <cassert>
#include <algorithm>


namespace coco{

Annotation::Annotation(const Value& value)
                      :id(value["id"].GetInt()),
                       image_id(value["image_id"].GetInt()),
                       category_id(value["category_id"].GetInt()),
                       area(value["area"].GetDouble()),
                       iscrowd(iscrowd)
{
  for(auto& polygon : value["segmentation"].GetArray()){
    std::vector<float> tmp;
    for(auto& coord : polygon.GetArray())
      tmp.push_back(coord.GetDouble());
    segmentation.push_back(tmp);
  }
  std::vector<std::vector<float>> segmentation;
  for(auto& bbox_value : value["bbox"].GetArray())
    bbox.push_back(bbox_value.GetDouble());
}

Image::Image(const Value& value)
            :id(value["id"].GetInt()),
             width(value["width"].GetInt()),
             height(value["height"].GetInt()),
             file_name(value["file_name"].GetString()){}

Categories::Categories(const Value& value)
                      :id(value["id"].GetInt()),
                       name(value["name"].GetString()),
                       supercategory(value["supercategory"].GetString()){}

COCO::COCO(const std::string annotation_file){
  std::ifstream ifs(annotation_file);
  IStreamWrapper isw(ifs);
  dataset.ParseStream(isw);
  assert(dataset.IsObject());
}

void COCO::CreateIndex(){
  assert(dataset.IsObject());
  if(dataset.HasMember("annotations")){
    for(auto& ann : dataset["annotations"].GetArray()){
      if(imgToAnns.count(ann["image_id"].GetInt())){ // if it exists
        imgToAnns[ann["image_id"].GetInt()].emplace_back(ann);
      }
      else{
        imgToAnns.insert(
          {ann["image_id"].GetInt(), std::vector<Annotation> {Annotation(ann)}}
        );
      }
      anns[ann["id"].GetInt()] = Annotation(ann);
    }
  }

  if(dataset.HasMember("images")){
    for(auto& img : dataset["images"].GetArray()){
      imgs[img["id"].GetInt()] = Image(img);
    }
  }

  if(dataset.HasMember("categories")){
    for(auto& cat : dataset["categories"].GetArray()){
      cats[cat["id"].GetInt()] = Categories(cat);
    }
  }

  if(dataset.HasMember("categories") && dataset.HasMember("annotations")){
    for(auto& ann : dataset["annotations"].GetArray()){
      if(catToImgs.count(ann["category_id"].GetInt())){
        catToImgs[ann["category_id"].GetInt()].push_back(ann["image_id"].GetInt());
      }
      else{
        catToImgs.insert(
          {ann["category_id"].GetInt(), std::vector<int> {ann["image_id"].GetInt()}}
        );
      }
    }
  }//ann and cat
}

std::vector<int> COCO::GetAnnIds(const std::vector<int> imgIds, 
                           const std::vector<int> catIds, 
                           const std::vector<float> areaRng, 
                           Crowd iscrowd)
{
  std::vector<int> returnAnns;
  std::vector<Annotation> tmp_anns;
  if(imgIds.size() == 0 && catIds.size() == 0 && areaRng.size() == 0){
    for(auto& ann : dataset["annotations"].GetArray()){
      tmp_anns.emplace_back(ann);
    }
  }
  else{
    if(imgIds.size() != 0){
      for(auto& imgId : imgIds){
        if(imgToAnns.count(imgId)){//if it exists
          tmp_anns.insert(tmp_anns.end(), imgToAnns[imgId].begin(), imgToAnns[imgId].end());
        }
      }
    }
    else{
      for(auto& ann : dataset["annotations"].GetArray()){
        tmp_anns.emplace_back(ann);
      }
    }

    if(catIds.size() != 0){
      for(auto it = tmp_anns.begin(); it != tmp_anns.end(); ++it){
        if(std::find(catIds.begin(), catIds.end(), it->category_id) == catIds.end()){
          it = tmp_anns.erase(it);
        }
      }
    }

    if(areaRng.size() != 0){
      for(auto it = tmp_anns.begin(); it != tmp_anns.end(); ++it){
        if(it->area <= areaRng[0] || it->area >= areaRng[1]){
          it = tmp_anns.erase(it);
        }
      }
    }
  }

  if(iscrowd == none){
    for(auto& i : tmp_anns){
      returnAnns.push_back(i.id);
    }
  }
  else{
    bool check = (iscrowd == F ? false : true);
    for(auto& i : tmp_anns){
      if(i.iscrowd == check)
        returnAnns.push_back(i.id);
    }
  }
  return returnAnns;
}

std::vector<int> COCO::GetCatIds(const std::vector<std::string> catNms, 
                                 const std::vector<std::string> supNms, 
                                 const std::vector<int> catIds)
{
  std::vector<int> returnIds;
  std::vector<Categories> cats;
  for(auto& cat: dataset["categories"].GetArray()){
    cats.emplace_back(cat);
  }
  if(catNms.size() != 0){
    for(auto it = cats.begin(); it != cats.end(); ++it){
      if(std::find(catNms.begin(), catNms.end(), it->name) == catNms.end())
        it = cats.erase(it);
    }
  }

  if(supNms.size() != 0){
    for(auto it = cats.begin(); it != cats.end(); ++it){
      if(std::find(supNms.begin(), supNms.end(), it->supercategory) == supNms.end())
        it = cats.erase(it);
    }
  }

  if(catIds.size() != 0){
    for(auto it = cats.begin(); it != cats.end(); ++it){
      if(std::find(catIds.begin(), catIds.end(), it->id) == catIds.end())
        it = cats.erase(it);
    }
  }
  for(auto& cat: cats){
    returnIds.push_back(cat.id);
  }

  return returnIds;
  
}

std::vector<Annotation> COCO::LoadAnns(std::vector<int> ids){
  std::vector<Annotation> returnAnns;
  for(auto& id : ids)
    returnAnns.push_back(anns[id]);

  return returnAnns;
}

std::vector<Image> COCO::LoadImgs(std::vector<int> ids){
  std::vector<Image> returnImgs;
  for(auto& id : ids)
    returnImgs.push_back(imgs[id]);

  return returnImgs;
}


}//coco namespace
