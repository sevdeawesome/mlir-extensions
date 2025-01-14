//===- DistOps.cpp - Dist dialect  ------------------------------*- C++ -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the Dist dialect and its basic operations.
///
//===----------------------------------------------------------------------===//

#include <imex/Dialect/Dist/IR/DistOps.h>
#include <imex/Dialect/Dist/Utils/Utils.h>
#include <imex/Dialect/PTensor/IR/PTensorOps.h>
#include <imex/Utils/PassUtils.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>

namespace imex {
namespace dist {

void DistDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include <imex/Dialect/Dist/IR/DistOpsTypes.cpp.inc>
      >();
  addOperations<
#define GET_OP_LIST
#include <imex/Dialect/Dist/IR/DistOps.cpp.inc>
      >();
}

} // namespace dist
} // namespace imex

#include <imex/Dialect/Dist/IR/DistOpsDialect.cpp.inc>
#define GET_TYPEDEF_CLASSES
#include <imex/Dialect/Dist/IR/DistOpsTypes.cpp.inc>
#define GET_OP_CLASSES
#include <imex/Dialect/Dist/IR/DistOps.cpp.inc>

namespace imex {
namespace dist {
::imex::ptensor::PTensorType getPTensorType(::mlir::Value t) {
  auto dtTyp = t.getType().dyn_cast<::imex::dist::DistTensorType>();
  if (dtTyp) {
    return dtTyp.getPTensorType();
  } else {
    return t.getType().dyn_cast<::imex::ptensor::PTensorType>();
  }
}
} // namespace dist
} // namespace imex
