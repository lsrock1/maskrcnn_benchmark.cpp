// std::map<string, torch::Tensor> saved;
// std::set<string> updated;
// std::map<string, string> mapping;

// void tmp_recur(shared_ptr<torch::jit::script::Module> module, std::string name){
//     string new_name;
//     if(name.compare("") != 0)
//       new_name = name + ".";
    
//     for(auto& u : module->get_parameters()){
//       torch::Tensor tensor = u.value().toTensor();
//       saved[new_name + u.name()] = tensor;
//     }
//     for(auto& u : module->get_attributes()){
//       torch::Tensor tensor = u.value().toTensor();
//       saved[new_name + u.name()] = tensor;
//     }
//     for(auto& i : module->get_modules())
//       tmp_recur(i, new_name + i->name());
//   }

{
    // modeling::GeneralizedRCNN model = modeling::BuildDetectionModel();

  //   torch::NoGradGuard guard;
  // auto body_module = torch::jit::load("../resource/r50fpnbackbone.pth");
  // tmp_recur(body_module, "");

  // auto box_extractor = torch::jit::load("../resource/r50fpnbox_extractor_fc6.pth");
  // for(auto& i : box_extractor->get_parameters()){
  //   if(i.name().find("weight") != std::string::npos)
  //     saved["roi_heads.box.feature_extractor.0.fc6.weight"] = i.value().toTensor();
  //   else
  //     saved["roi_heads.box.feature_extractor.0.fc6.bias"]= i.value().toTensor();
  // }

  //  box_extractor = torch::jit::load("../resource/r50fpnbox_extractor_fc7.pth");
  // for(auto& i : box_extractor->get_parameters()){
  //   if(i.name().find("weight") != std::string::npos)
  //     saved["roi_heads.box.feature_extractor.0.fc7.weight"] = i.value().toTensor();
  //   else
  //     saved["roi_heads.box.feature_extractor.0.fc7.bias"]= i.value().toTensor();
  // }

  //  box_extractor = torch::jit::load("../resource/r50fpnbox_predictor_bbox.pth");
  // for(auto& i : box_extractor->get_parameters()){
  //   if(i.name().find("weight") != std::string::npos)
  //     saved["roi_heads.box.predictor.0.bbox_pred.weight"] = i.value().toTensor();
  //   else
  //     saved["roi_heads.box.predictor.0.bbox_pred.bias"]= i.value().toTensor();
  // }

  //  box_extractor = torch::jit::load("../resource/r50fpnbox_predictor_cls.pth");
  // for(auto& i : box_extractor->get_parameters()){
  //   if(i.name().find("weight") != std::string::npos)
  //     saved["roi_heads.box.predictor.0.cls_score.weight"] = i.value().toTensor();
  //   else{
  //     saved["roi_heads.box.predictor.0.cls_score.bias"]= i.value().toTensor();
  //   }
  // }

  //  box_extractor = torch::jit::load("../resource/r50fpnrpn_bbox_pred.pth");
  // for(auto& i : box_extractor->get_parameters()){
  //   if(i.name().find("weight") != std::string::npos)
  //     saved["rpn.rpnhead.bbox_pred.weight"] = i.value().toTensor();
  //   else
  //     saved["rpn.rpnhead.bbox_pred.bias"]= i.value().toTensor();
  // }

  //  box_extractor = torch::jit::load("../resource/r50fpnrpn_cls_logits.pth");
  // for(auto& i : box_extractor->get_parameters()){
  //   if(i.name().find("weight") != std::string::npos)
  //     saved["rpn.rpnhead.cls_logits.weight"] = i.value().toTensor();
  //   else
  //     saved["rpn.rpnhead.cls_logits.bias"]= i.value().toTensor();
  // }

  //  box_extractor = torch::jit::load("../resource/r50fpnrpn_conv.pth");
  // for(auto& i : box_extractor->get_parameters()){
  //   if(i.name().find("weight") != std::string::npos)
  //     saved["rpn.rpnhead.conv.weight"] = i.value().toTensor();
  //   else
  //     saved["rpn.rpnhead.conv.bias"]= i.value().toTensor();
  // }
  // // saved["roi_heads.box.feature_extractor.0.fc6.weight"] = box_extractor.get_parameter("weight");
  // // for(auto i = saved.begin(); i != saved.end(); ++i)
  // //   cout << i->first << "\n";

  // for(auto& i : model->named_parameters()){
  //   if(i.key().find("fpn") != std::string::npos){
  //     std::string n = i.key().substr(20);
  //     n = n.substr(0, 14);
  //     if(i.key().find("weight") != std::string::npos){
  //       n = n + ".weight";
  //     }
  //     else{
  //       n = n + ".bias";
  //     }
  //     std::cout << i.key() << "\n";
  //     assert(saved.count(n));
  //     i.value().copy_(saved.at(n));
  //     updated.insert(i.key());
  //     mapping[i.key()] = n;
  //   }
  //   else if(i.key().find("backbone") != std::string::npos){
  //     std::string n = "body" + i.key().substr(19);
  //     std::cout << i.key() <<"\n";
  //     assert(saved.count(n));
  //     i.value().copy_(saved.at(n));
  //     updated.insert(i.key());
  //     mapping[i.key()] = n;
  //   }
  //   else{
  //     std::cout << i.key() << "\n";
  //     assert(saved.count(i.key()));
  //     i.value().copy_(saved.at(i.key()));
  //     updated.insert(i.key());
  //     mapping[i.key()] = i.key();
  //   }
  //   // else{
  //   //   cout << i.key() << "\n";
  //   //   assert(false);
  //   // }
  // }

  // for(auto& i : model->named_buffers()){
  //   if(i.key().find("fpn") != std::string::npos){
  //     std::string n = i.key().substr(20);
  //     n = n.substr(0, 14);
  //     if(i.key().find("weight") != std::string::npos){
  //       n = n + ".weight";
  //     }
  //     else{
  //       n = n + ".bias";
  //     }
  //     std::cout << i.key() << "\n";
  //     assert(saved.count(n));
  //     i.value().copy_(saved.at(n));
  //     updated.insert(i.key());
  //     mapping[i.key()] = n;
  //   }
  //   else if(i.key().find("backbone") != std::string::npos){
  //     std::string n = "body" + i.key().substr(19);
  //     std::cout << i.key() << "\n";
  //     assert(saved.count(n));
  //     i.value().copy_(saved.at(n));
  //     updated.insert(i.key());
  //     mapping[i.key()] = n;
  //   }
  //   else if(i.key().find("anchor_generator") != std::string::npos){
  //     updated.insert(i.key());
  //     mapping[i.key()] = i.key();
  //   }
  //   else{
  //     std::cout << i.key() << "\n";
  //     assert(saved.count(i.key()));
  //     i.value().copy_(saved.at(i.key()));
  //     updated.insert(i.key());
  //     mapping[i.key()] = i.key();
  //   }
  //   // else{
  //   //   cout << i.key() << "\n";
  //   //   assert(false);
  //   // }
  // }
  //   torch::serialize::OutputArchive archive;
  // for(auto& i : model->named_parameters()){
  //   std::cout << "saved name : " << i.key() << "\n";
  //   assert(updated.count(i.key()));
  //   assert( (saved.at(mapping[i.key()]) != i.value()).sum().item<int>() == 0);
  //   archive.write(i.key(), i.value());
  // }
  // for(auto& i : model->named_buffers()){
  //   std::cout << "saved buffer name : " << i.key() << "\n";
  //   assert(updated.count(i.key()));
  //   if(i.key().find("anchor_generator") == std::string::npos)
  //     assert( (saved.at(mapping[i.key()]) != i.value()).sum().item<int>() == 0);
  //   archive.write(i.key(), i.value(), true);
  // }
  // archive.save_to("../resource/frcn_r50_cpp.pth");
  
  // torch::NoGradGuard guard;
  // auto body = modeling::ResNet();

  // auto module_ = torch::jit::load("../resource/resnet101.pth");
  // tmp_recur(module_, "");
  // for(auto i = saved.begin(); i != saved.end(); ++i)
  //   cout << i->first << "\n";
  
  // for(auto& i : body->named_parameters()){
  //   cout << i.key() << "\n";
  //   if(i.key().find("stem") != string::npos)
  //     i.value().copy_(saved.at(i.key().substr(5)));
  //   else
  //     i.value().copy_(saved.at(i.key()));
  // }
  // for(auto& i : body->named_buffers()){
  //   cout << i.key() << "\n";    
  //   if(i.key().find("stem") != string::npos)
  //     i.value().copy_(saved.at(i.key().substr(5)));
  //   else
  //     i.value().copy_(saved.at(i.key()));
  // }
  
  //   torch::serialize::OutputArchive archive;
  // for(auto& i : body->named_parameters())
  //   archive.write(i.key(), i.value());
  // for(auto& i : body->named_buffers()){
  //   std::cout << "saved buffer name : " << i.key() << "\n";
  //   archive.write(i.key(), i.value(), true);
  // }
  // archive.save_to("../resource/resnet101_cpp.pth");
}