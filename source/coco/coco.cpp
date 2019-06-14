#include "coco.h"
#include <rapidjson/istreamwrapper.h>
#include <fstream>
#include <cassert>
#include <algorithm>
#include <cstring>
#include <iostream>


namespace coco{

Annotation::Annotation(const Value& value)
                      :id(value["id"].GetDouble()),
                       image_id(value["image_id"].GetInt()),
                       category_id(value["category_id"].GetInt()),
                       area(value["area"].GetDouble()),
                       iscrowd(value["iscrowd"].GetInt())
{
  if(iscrowd){
    for(auto& coord : value["segmentation"]["counts"].GetArray())
      counts.push_back(coord.GetInt());
    size = std::make_pair(value["segmentation"]["size"][0].GetInt(), value["segmentation"]["size"][1].GetInt());
  }
  else{
    for(auto& polygon : value["segmentation"].GetArray()){
      std::vector<double> tmp;
      for(auto& coord : polygon.GetArray())
        tmp.push_back(coord.GetDouble());
      segmentation.push_back(tmp);
    }
  }
  
  for(auto& bbox_value : value["bbox"].GetArray())
    bbox.push_back(bbox_value.GetDouble());
}

Annotation::Annotation(): id(0), image_id(0), category_id(0), area(0), iscrowd(0){}

Image::Image(const Value& value)
            :id(value["id"].GetInt()),
             width(value["width"].GetInt()),
             height(value["height"].GetInt()),
             file_name(value["file_name"].GetString()){}
             //file_name(strdup(value["file_name"].GetString())){}

// Image::~Image(){
//   if(file_name)
//     delete[] file_name;
// }

Image::Image(): id(0), width(0), height(0), file_name(""){}

// Image::Image(const Image& other){
//   id = other.id;
//   width = other.width;
//   height = other.height;
//   if(file_name)
//     delete[] file_name;
//   file_name = new char[strlen(other.file_name) + 1];
//   strcpy(file_name, other.file_name);
// }

// Image::Image(Image&& other){
//   id = other.id;
//   width = other.width;
//   height = other.height;
//   if(file_name)
//     delete[] file_name;
//   file_name = other.file_name;
//   other.file_name = nullptr;
// }

// Image& Image::operator=(const Image& other){
//   if(this != &other){
//     id = other.id;
//     width = other.width;
//     height = other.height;
//     if(file_name)
//       delete[] file_name;
//     file_name = new char[strlen(other.file_name)+1];
//     strcpy(file_name, other.file_name);
//   }
//   return *this;
// }

// Image& Image::operator=(Image&& other){
//   if(this != &other){
//     id = other.id;
//     width = other.width;
//     height = other.height;
//     if(file_name)
//       delete[] file_name;
//     file_name = other.file_name;
//     other.file_name = nullptr;
//   }
//   return *this;
// }

Categories::Categories(const Value& value)
                      :id(value["id"].GetInt()),
                       name(strdup(value["name"].GetString())),
                       supercategory(strdup(value["supercategory"].GetString())){}

Categories::~Categories(){
  if(supercategory)
    delete[] supercategory;
  if(name)
    delete[] name;
}

Categories::Categories(): id(0), name(nullptr), supercategory(nullptr){}

Categories::Categories(const Categories& other){
  id = other.id;
  if(name)
    delete[] name;
  name = new char[strlen(other.name) + 1];
  if(supercategory)
    delete[] supercategory;
  supercategory = new char[strlen(other.supercategory) + 1];
  strcpy(name, other.name);
  strcpy(supercategory, other.supercategory);
}

Categories::Categories(Categories&& other){
  id = other.id;
  if(name)
    delete[] name;
  name = other.name;
  other.name = nullptr;
  if(supercategory)
    delete[] supercategory;
  supercategory = other.supercategory;
  other.supercategory = nullptr;
}

Categories& Categories::operator=(const Categories& other){
  if(this != &other){
    id = other.id;
    if(name)
      delete[] name;
    name = new char[strlen(other.name) + 1];
    strcpy(name, other.name);
    if(supercategory)
      delete[] supercategory;
    supercategory = new char[strlen(other.supercategory) + 1];
    strcpy(supercategory, other.supercategory);
  }
  return *this;
}

Categories& Categories::operator=(Categories&& other){
  if(this != &other){
    id = other.id;
    if(name)
      delete[] name;
    name = other.name;
    other.name = nullptr;
    if(supercategory)
      delete[] supercategory;
    supercategory = other.supercategory;
    other.supercategory = nullptr;
  }
  return *this;
}

COCO::COCO(std::string annotation_file){
  std::ifstream ifs(annotation_file);
  IStreamWrapper isw(ifs);
  dataset.ParseStream(isw);
  assert(dataset.IsObject());
  CreateIndex();
}

void COCO::CreateIndex(){
  if(dataset.HasMember("annotations")){
    for(auto& ann : dataset["annotations"].GetArray()){
      if(imgToAnns.count(ann["image_id"].GetInt())){ // if it exists
        imgToAnns[ann["image_id"].GetInt()].emplace_back(ann);
      }
      else{
        imgToAnns[ann["image_id"].GetInt()] = std::vector<Annotation> {Annotation(ann)};
      }
      anns[static_cast<int64_t>(ann["id"].GetDouble())] = Annotation(ann);
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

std::vector<int64_t> COCO::GetAnnIds(const std::vector<int> imgIds, 
                           const std::vector<int> catIds, 
                           const std::vector<float> areaRng, 
                           Crowd iscrowd)
{
  std::vector<int64_t> returnAnns;
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

std::vector<Annotation> COCO::LoadAnns(std::vector<int64_t> ids){
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
