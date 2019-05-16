#include "gtest/gtest.h"

#include <defaults.h>

using namespace rcnn::config;

TEST(defaults, SetCFGFromFile)
{
    CFGString model_weight = GetCFG<CFGString>({"MODEL", "WEIGHT"});
    //check if default empty, yaml exists
    EXPECT_STREQ("catalog://ImageNetPretrained/MSRA/R-50", model_weight.get());

    // double lr = GetCFG<double>({"SOLVER", "BASE_LR"});
    // //check if default exists, yaml exists
    // EXPECT_EQ(lr, 0.01);
    
    // bool stride1x1 = GetCFG<bool>({"MODEL", "RESNETS", "STRIDE_IN_1X1"});
    // //check if default exists, yaml empty
    // EXPECT_EQ(stride1x1, true);
}

// TEST(AccountInfo, DeckControl)
// {
//     AccountInfo player;

//     EXPECT_NO_THROW(player.ShowDeckList());
//     EXPECT_EQ(false, player.CreateDeck("deck1", CardClass::INVALID));

//     player.CreateDeck("deck2", CardClass::DREAM);
//     player.CreateDeck("deck3", CardClass::DRUID);

//     EXPECT_EQ(2, static_cast<int>(player.GetNumOfDeck()));
//     EXPECT_NO_THROW(player.ShowDeckList());

//     EXPECT_NO_THROW(player.DeleteDeck(0));
//     EXPECT_EQ(1, static_cast<int>(player.GetNumOfDeck()));
//     EXPECT_EQ("deck3", player.GetDeck(0)->GetName());

//     EXPECT_EQ(1, static_cast<int>(player.GetNumOfDeck()));
// }