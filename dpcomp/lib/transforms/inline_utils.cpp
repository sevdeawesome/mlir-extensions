// Copyright 2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "plier/transforms/inline_utils.hpp"

#include "plier/dialect.hpp"

#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/InliningUtils.h>

namespace {
static bool mustInline(mlir::CallOp call, mlir::FuncOp func) {
  auto attr = mlir::StringAttr::get(plier::attributes::getForceInlineName(),
                                    call.getContext());
  return call->hasAttr(attr) || func->hasAttr(attr);
}

struct ForceInline : public mlir::OpRewritePattern<mlir::CallOp> {
  using mlir::OpRewritePattern<mlir::CallOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::CallOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto mod = op->getParentOfType<mlir::ModuleOp>();
    assert(mod);

    auto func = mod.lookupSymbol<mlir::FuncOp>(op.getCallee());
    if (!func)
      return mlir::failure();

    if (!mustInline(op, func))
      return mlir::failure();

    auto loc = op.getLoc();
    auto reg =
        rewriter.create<mlir::scf::ExecuteRegionOp>(loc, op.getResultTypes());
    auto newCall = [&]() -> mlir::Operation * {
      auto &regBlock = reg.region().emplaceBlock();
      mlir::OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToStart(&regBlock);
      auto call = rewriter.clone(*op);
      rewriter.create<mlir::scf::YieldOp>(loc, call->getResults());
      return call;
    }();

    mlir::InlinerInterface inlinerInterface(op->getContext());
    auto parent = op->getParentOp();
    rewriter.startRootUpdate(parent);
    auto res =
        mlir::inlineCall(inlinerInterface, newCall, func, &func.getRegion());
    if (mlir::succeeded(res)) {
      assert(newCall->getUsers().empty());
      rewriter.eraseOp(newCall);
      rewriter.replaceOp(op, reg.getResults());
      rewriter.finalizeRootUpdate(parent);
    } else {
      rewriter.eraseOp(reg);
      rewriter.cancelRootUpdate(parent);
    }
    return res;
  }
};

struct ForceInlinePass
    : public mlir::PassWrapper<ForceInlinePass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::StandardOpsDialect>();
    registry.insert<mlir::scf::SCFDialect>();
  }

  virtual mlir::LogicalResult initialize(mlir::MLIRContext *context) override {
    mlir::OwningRewritePatternList p(context);
    p.insert<ForceInline>(context);
    patterns = std::move(p);
    return mlir::success();
  }

  virtual void runOnOperation() override {
    auto mod = getOperation();
    (void)mlir::applyPatternsAndFoldGreedily(mod, patterns);

    mod->walk([&](mlir::CallOp call) {
      auto func = mod.lookupSymbol<mlir::FuncOp>(call.getCallee());
      if (func && mustInline(call, func)) {
        call.emitError("Couldn't inline force-inline call");
        signalPassFailure();
      }
    });
  }

private:
  mlir::FrozenRewritePatternSet patterns;
};
} // namespace

std::unique_ptr<mlir::Pass> plier::createForceInlinePass() {
  return std::make_unique<ForceInlinePass>();
}