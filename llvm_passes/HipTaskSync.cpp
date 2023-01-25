//===- HipTaskSync.cpp ---------------------------------------------===//
//
// Part of the CHIP-SPV Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// A pass to handle HIP cooperative group synchronization related operations.
//
//===----------------------------------------------------------------------===//

#include "HipTaskSync.h"

#include "LLVMSPIRV.h"
#include "../src/common.hh"

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include <map>
#include <string>
#include <iostream>

using namespace std;

#define PASS_ID "hip-function-ptr"
#define DEBUG_TYPE PASS_ID

using namespace llvm;

namespace {

namespace CGSyncCodePattern {
  enum {
    Unsolved = 0,
    Simple = 1, 
    InBranchBody = 2, 
    InLoopBody = 3, 
    Unsupported = 4
  };
} // CGSyncCodePattern

static bool isKernelCall(Instruction *I) {
  return true;
}


// if (cond) { }
//    CJ_S1;
//    cg.sync();
//    CJ_S2;
// } else {
//    S3;
// }





//    S3; 
//    cg.sync();
//    S4; 
//    cg.sync();
//    S5;
// }

class CooperativeGroupAnalyzer {
protected:
  // The call isntructions related to cooperative groups
  map<Instruction*, unsigned> CGSyncCalls;
  // The scope of cooperative group synchronization
  map<Instruction*, SmallVector<BasicBlock* , 2> > CGRegions;

public:
  CooperativeGroupAnalyzer() {};

  // Collect cooperative group calls and reconstruct the CG related loops
  bool handleKernelFunction(Function* F) {
    // Identify the start point for initial region, this should be the position that the cooperative 
    // group object was created  
    Value* TIDVal = nullptr;
    Value* GroupNumVal = nullptr;
    Instruction* CGInitInst = IdentifyCGInit(F->getEntryBlock(), &TIDVal, &GroupNumVal);
    if (CGInitInst == nullptr)
      // The entry basic block does not contain cooperative group object creation
      return false;

    map<int, SmallVector<BasicBlock *, 16> > Paths;
    SmallVector<Instruction* , 16> CGSyncPath;
    // Collect the path that contains cooperative group sync calls, if there are more than one path, 
    // then the transformation can not go through
    if (!CollectCGSyncCalls(F, Paths, CGSyncPath))
      return false;

    // Verify the legality of transformation 
    for (Instruction* I : CGSyncPath) {
      /*CGSyncCodePattern CodePattern = VerifyCGTrans(I, DT, LI);
      if (CodePattern == CGSyncCodePattern::Unsupported)
        // If the cooperative sync call can not be handled, then give up the transformation for the whole function
        return false;
      else
	CGSyncCalls[I] = CodePattern;
      */
    }

    // Create the region of cooperative synchronization
    CreateCGRegions(CGSyncPath, CGInitInst);

    // Apply the actual transformation
    for (auto pair : CGRegions)  
      ApplyTransformation(pair.first, TIDVal, GroupNumVal, pair.second);
  }

protected:
  // Collect all cooperative group sync calls in the order of actual execution path, ant there should be 
  // only one path
  bool CollectCGSyncCalls(Function* F,  map<int, SmallVector<BasicBlock *, 16> >& Paths, 
			  SmallVector<Instruction* , 16>& CGSyncPath) {
    // Check through the function to collect the cooperate group sync operation
    SmallPtrSet<BasicBlock *, 16> Visited;
    BasicBlock& EntryBB = F->getEntryBlock();
    int PathCount = 0;
    // Register 1st path
    Paths[0].push_back(&EntryBB);
    // Visit control flow path
    PathCount = VisitCFGPaths(&EntryBB, PathCount, Paths, Visited);

    // Collect sync calls
    map<BasicBlock* , SmallVector<Instruction *, 16> > BB2CGSyncs;
    for (auto &BB : * F)
      for (auto &I : BB) {
	if (!IsCGSyncCall(&I)) {
	  // Register CGSyncCall status
          CGSyncCalls[&I] = CGSyncCodePattern::Unsolved;
          // Register CGSyncCall with basic blocks
          BB2CGSyncs[&BB].push_back(&I);
        }
      }
    
    // Check through paths, only one path can be feasible for transformation
    if (Paths.size() == 0)
      return false;

    SmallVector<Instruction* , 16> CGSyncInsts;
    for (auto pair : Paths) {
      SmallVector<BasicBlock *, 16>& Path = pair.second;
      
      Visited.clear();
      CGSyncInsts.clear();
      // Check if the path contains all CGSync related basic blocks
      for (BasicBlock * BB : Path) {
        if (BB2CGSyncs.find(BB) != BB2CGSyncs.end()) {
          // Register the CGSync basic block
          Visited.insert(BB);
          // Collect CGSync instruction in execution order
          CGSyncInsts = BB2CGSyncs[BB];
        }
      }

      // Check if the visited basic blocks match actuall number of CGSync basic blocks
      if (Visited.size() == BB2CGSyncs.size()) {
        // Put the collected sequence of CGSync instructions into CGSync path
        if (CGSyncPath.size() == 0)
          CGSyncPath = CGSyncInsts;
        else if (CGSyncPath.size() != CGSyncInsts.size()) {
          // This should not happen
          CGSyncPath.clear();

	  return false;
	}
    } else if (Visited.size() == 0) {
      // The path that does not contain any CGSync call is allowed
      continue;
    } else {
      // There is path that does not match all CGSync calls
      CGSyncPath.clear();

	    return false;
    }
  }

    return true;
  }

  // Visit control flow  path
  int VisitCFGPaths(BasicBlock* BB, int PathID, map<int, SmallVector<BasicBlock *, 16> >& Paths, 
		    SmallPtrSet<BasicBlock *, 16>& Visited) {
    if (Visited.contains(BB))
      return PathID;
    
    // Register visited node
    Visited.insert(BB);

    // Check through successors
    Instruction* TermInst = BB->getTerminator();
    if (!TermInst)
      return PathID;

    // No successor
    if (TermInst->getNumSuccessors() == 0)
      return PathID;

    // Preserve path 
    SmallVector<BasicBlock *, 16> CurrPath;
    CurrPath = Paths[PathID];

    int PathCount = PathID;
    for (int i = 0; i < TermInst->getNumSuccessors(); i ++) {
      if (i > 0) {
        // Replicate path
        Paths[PathCount] = CurrPath;
            
        // Increment path count
        PathCount ++;
      }
      
      BasicBlock* SuccBB = TermInst->getSuccessor(i);
      // Register basic block into current path
      Paths[PathCount].push_back(SuccBB);
      // Visit control flow path
      PathCount = VisitCFGPaths(SuccBB, PathCount, Paths, Visited);
    }

    return PathCount;
  }

  // Check if the given call instruction is a cooperative group symchronization related calls
  bool IsCGSyncCall(Instruction* I) {
    if (auto *CI = dyn_cast<CallInst>(I)) {
      if (auto *F = CI->getCalledFunction()) {
        // TODO: check 1st argument's type 
        
        string FuncNameStr = F->getName().data();
        if (FuncNameStr.find("sync", 1) > 0)
          return true;
      }
    }

    return false;
  }

  // Check if the given call instruction is a getting thread rank operation
  bool IsGetThreadRank(Instruction* I) {
    if (auto *CI = dyn_cast<CallInst>(I)) {
      if (auto *F = CI->getCalledFunction()) {
	      // TODO: check 1st argument's type

        string FuncNameStr = F->getName().data();
        if (FuncNameStr.find("thread_rank", 1) > 0)
          return true;
      }
    }

    return false;
  }

  // Check if the call instruction is to call cg.size
  bool IsGroupNum(Instruction* I) {
    if (auto *CI = dyn_cast<CallInst>(I)) {
      if (auto *F = CI->getCalledFunction()) {
	      // TODO: check 1st argument's type 

	      string FuncNameStr = F->getName().data();
	      if (FuncNameStr.find("size", 1) > 0)
	        return true;
      }
    }

    return false;
  }
    
  // Check the code pattern for cooperative sync call, and work out the code pattern for further transformation
  unsigned VerifyCGTrans(Instruction* I, DominatorTree& DT, LoopInfo& LI) {
    // TODO: check code pattern
    
    return CGSyncCodePattern::Unsupported;
  }
  
  // Create the cooperative group sync regions
  bool CreateCGRegions(SmallVector<Instruction *, 16>& CGSyncPath, Instruction* CGInitInst) {
    bool Result = true;

    // Splict the initial region
    BasicBlock* InitBB = CGInitInst->getParent();
    BasicBlock* PredBB = InitBB->splitBasicBlockBefore(CGInitInst);

    CGRegions[CGInitInst].push_back(InitBB);
    
    Instruction* LastInst = CGInitInst;

    // Splict basic blocks
    for (Instruction* I : CGSyncPath) {
      BasicBlock* BB = I->getParent();
      PredBB = BB->splitBasicBlockBefore(I);
      
      // Add the end basic block for last region
      CGRegions[LastInst].push_back(PredBB);

      // Add the start basic block for current region
      CGRegions[I].push_back(PredBB);
      
      LastInst = I;
    }

    return Result;
  }

  // Identify the initial position of the first region, return thread rank value (it is presented as a 
  // call instruction)
  Instruction* IdentifyCGInit(BasicBlock& BB, Value** TIDPtr, Value** GroupNumPtr) {
    Instruction* RetVal = nullptr;
    for (Instruction& I : BB) {
      // if (IsCGObjCreation(&I)) {
      if (IsGetThreadRank(&I)) {
	RetVal = &I;
	* TIDPtr = &I;
      } else if (IsGroupNum(&I)) {
	* GroupNumPtr = &I;
      }
    }
    
    return RetVal;
  }
  
  // Create the iterator initialization , i = 0
  Value* CreateIterator(BasicBlock* PredBB) {
    IRBuilder<> Builder(PredBB);
    
    // Create iterator
    Value* AllocIter = Builder.CreateAlloca(Type::getInt32Ty(PredBB->getContext())); // i
    Value* Const0 = Builder.getInt32(0); 
    Builder.CreateStore(Const0, AllocIter); // i = 0
    
  }

  // Create prologue for cooperative group execution
  Value* CreatePrologue(BasicBlock* PredBB, BasicBlock* CurrBB, 
			BasicBlock* LoopHeaderBB, BasicBlock* LoopTailBB, 
			Value* TIDVal, Value* GroupNumVal, Value* InitIterVal) {
    IRBuilder<> Builder(LoopHeaderBB);
    // Create phi node 
    PHINode* PhiVal = Builder.CreatePHI(Type::getInt32Ty(PredBB->getContext()), 2);
    PhiVal->setIncomingValue(0, InitIterVal);
    PhiVal->setIncomingBlock(0, PredBB);

    // Add modular instruction
    Value* ModVal = Builder.CreateAnd(TIDVal, GroupNumVal);
    Value* CondVal = Builder.CreateICmpEQ(ModVal, 
					  ConstantInt::get(Type::getInt32Ty(PredBB->getContext()), 0), 
					  "TIDCmp"); // ModVal == 0 
    Builder.CreateCondBr(CondVal, CurrBB, LoopTailBB); // If (ModVal == 0) goto CurrBB else goto LoopTailBB

    return PhiVal;
  }

  // Create the epilogue for cooperative group execution
  void CreateEpilogue(BasicBlock* SuccBB, BasicBlock* LoopHeaderBB, BasicBlock* LoopTailBB, 
		      Value* BIterVal, Value* GroupNumVal) {
    IRBuilder<> Builder(LoopTailBB);

    PHINode* PhiVal = dyn_cast<PHINode>(BIterVal);
    assert(PhiVal && "Must be PHINode");

    // Append the iterator increment operation
    Value* IterVal = Builder.CreateAdd(PhiVal, 
				       ConstantInt::get(Type::getInt32Ty(SuccBB->getContext()), 1));  // i = i + 1
    // Setup phi node related predecessor
    PhiVal->setIncomingValue(1, IterVal);
    PhiVal->setIncomingBlock(1, LoopTailBB);

    // Append condition branch instruction
    Value* CondVal = Builder.CreateICmpUGE(IterVal, GroupNumVal, "IterCmp"); // i >= GroupNum 
    Builder.CreateCondBr(CondVal, SuccBB, LoopHeaderBB); // if (i >= GroupNum ) then goto SuccBB else goto LoopHeaderBB
  }

  // Apply cooperative group relatd code transformation
  bool ApplyTransformation(Instruction* I, Value* TIDVal, Value* GroupNumVal, 
			   SmallVector<BasicBlock* , 2> CGRegion) {
    // Get the basic blocks for region scope
    // THe predecessor of CG region
    BasicBlock* PredBB = CGRegion[0];
    // The successor of CG region
    BasicBlock* SuccBB = CGRegion[1];
    // The 1st basic block of CG region
    BasicBlock* CurrBB = I->getParent();
    // Create basic blocks for loop header and tail
    BasicBlock* LoopHeaderBB = CurrBB->splitBasicBlockBefore(I);
    // BasicBlock::Create(PredBB->getContext(), "", PredBB->getParent(), PredBB);
    BasicBlock* LoopTailBB = SuccBB->getSinglePredecessor();

    // Create prelogue and epilogue for region
    // Create initialization of iterator
    Value* InitIterVal = CreateIterator(PredBB); // for (i = 0; i < GroupNum; i ++)
    
    // Create prologue 
    Value* PhiVal = CreatePrologue(PredBB, CurrBB, LoopHeaderBB, LoopTailBB, TIDVal, GroupNumVal, 
				   InitIterVal);
    
    // Create epiogue  
    CreateEpilogue(SuccBB, LoopHeaderBB, LoopTailBB, PhiVal, GroupNumVal);

    return true;
  }
};

// Create a pointer type to to named opaque struct.
static Type *getPointerTypeToOpaqueStruct(LLVMContext &C, StringRef Name,
                                          unsigned AddrSpace = 0) {
  Type *Ty = StructType::getTypeByName(C, Name);
  if (!Ty)
    Ty = StructType::create(C, Name);
  return Ty->getPointerTo(AddrSpace);
}

// Create a temporary value definition (an instruction) which is intended to be
// replaced later with an actual definition.
static Instruction *getPlaceholder(Type *Ty, Function *F) {
  // Use the freeze instruction from a poison value as a temporary definition.
  auto *PV = PoisonValue::get(Ty);
  return new FreezeInst(PV, "placeholder",
                        F->getEntryBlock().getFirstNonPHIOrDbg());
}

static bool handleTaskSync(Module &M) {
  bool Changed = true;
  
  SmallPtrSet<Function *, 16> Worklist;
  for (auto &F : M) {
    // Collect kernel functions 
    if (F.getCallingConv() == CallingConv::SPIR_KERNEL) {
      Worklist.insert(&F);
      continue;
    }
  }

  CooperativeGroupAnalyzer CGAnalyzer;

  for (auto *F : Worklist) {
    dbgs() << "Task sync in " << F->getName() << " \n";
  
    string funcName = F->getName().data();
    std::cout << "Func name: " << funcName << std::endl;
  
    CGAnalyzer.handleKernelFunction(F);
  }

  return Changed;
}

} // namespace

PreservedAnalyses HipTaskSyncPass::run(Module &M,
                                              ModuleAnalysisManager &AM) {
  return handleTaskSync(M) ? PreservedAnalyses::none()
    : PreservedAnalyses::all();
}

extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, PASS_ID, LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, ModulePassManager &MPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name == PASS_ID) {
                    MPM.addPass(HipTaskSyncPass());
                    return true;
                  }
                  return false;
                });
          }};
}
