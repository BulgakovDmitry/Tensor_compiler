#include <gtest/gtest.h>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "tensor.h"

using namespace tensor_compiler;

// ------------------------------ Helpers ----------------------------------------

static std::vector<float> unpackFloatsFromRaw(const std::string& raw) {
    EXPECT_EQ(raw.size() % sizeof(float), 0u);

    std::vector<float> out(raw.size() / sizeof(float));
    if (!raw.empty()) {
        std::memcpy(out.data(), raw.data(), raw.size());
    }
    return out;
}

static dim_type makeDim(std::initializer_list<int64_t> dims) {
    dim_type d;
    for (auto v : dims) d.Add(v);
    return d;
}

// ----------------------------- Constructors ------------------------------------

TEST(Tensor, DefaultConstructorHasExpectedDefaults) {
    Tensor t;

    EXPECT_TRUE(t.name().empty());
    EXPECT_EQ(t.type(), data_type::TensorProto_DataType_UNDEFINED);
    EXPECT_EQ(t.kind(), Tensor_kind::unknown);

    EXPECT_TRUE(t.data().empty());
    EXPECT_TRUE(t.shape().empty());

    auto dim = t.dim();
    EXPECT_EQ(dim.size(), 0);
}

TEST(Tensor, ParameterizedConstructorStoresFields) {
    std::string name = "W";
    std::vector<int64_t> shape = {2, 3};
    std::string data = "rawbytes";
    Tensor_kind kind = Tensor_kind::constant;

    Tensor t(name, data_type::TensorProto_DataType_FLOAT, shape, data, kind);

    EXPECT_EQ(t.name(), "W");
    EXPECT_EQ(t.type(), data_type::TensorProto_DataType_FLOAT);
    EXPECT_EQ(t.kind(), Tensor_kind::constant);
    EXPECT_EQ(t.data(), "rawbytes");

    ASSERT_EQ(t.shape().size(), 2u);
    EXPECT_EQ(t.shape()[0], 2);
    EXPECT_EQ(t.shape()[1], 3);

    auto dim = t.dim();
    EXPECT_EQ(dim.size(), 0);
}

// ------------------------------- create ----------------------------------------

TEST(Tensor, CreateStoresNameShapeKindAndFloatType) {
    std::string name = "X";
    std::vector<int64_t> shape = {1, 4};
    std::vector<float> data = {1.0f, 2.0f, 3.5f, -4.25f};

    Tensor t = Tensor::create(name, shape, data, Tensor_kind::input);

    EXPECT_EQ(t.name(), "X");
    EXPECT_EQ(t.kind(), Tensor_kind::input);
    EXPECT_EQ(t.type(), data_type::TensorProto_DataType_FLOAT);

    ASSERT_EQ(t.shape().size(), 2u);
    EXPECT_EQ(t.shape()[0], 1);
    EXPECT_EQ(t.shape()[1], 4);

    EXPECT_EQ(t.data().size(), data.size() * sizeof(float));

    auto unpacked = unpackFloatsFromRaw(t.data());
    ASSERT_EQ(unpacked.size(), data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        EXPECT_FLOAT_EQ(unpacked[i], data[i]);
    }
}

TEST(Tensor, CreateWithEmptyDataProducesEmptyRawData) {
    Tensor t = Tensor::create("Empty", {2, 2}, {}, Tensor_kind::intermediate);

    EXPECT_EQ(t.name(), "Empty");
    EXPECT_EQ(t.kind(), Tensor_kind::intermediate);
    EXPECT_EQ(t.type(), data_type::TensorProto_DataType_FLOAT);

    ASSERT_EQ(t.shape().size(), 2u);
    EXPECT_EQ(t.shape()[0], 2);
    EXPECT_EQ(t.shape()[1], 2);

    EXPECT_TRUE(t.data().empty());
}

// ------------------------------- Getters ---------------------------------------

TEST(Tensor, GettersReturnSetValues) {
    Tensor t;

    t.setName("T");
    t.setType(data_type::TensorProto_DataType_INT64);
    t.setKind(Tensor_kind::output);
    t.setData("abc");
    t.setShape({10, 20});
    t.setDim(makeDim({10, 20}));

    EXPECT_EQ(t.name(), "T");
    EXPECT_EQ(t.type(), data_type::TensorProto_DataType_INT64);
    EXPECT_EQ(t.kind(), Tensor_kind::output);
    EXPECT_EQ(t.data(), "abc");

    ASSERT_EQ(t.shape().size(), 2u);
    EXPECT_EQ(t.shape()[0], 10);
    EXPECT_EQ(t.shape()[1], 20);

    auto dim = t.dim();
    ASSERT_EQ(dim.size(), 2);
    EXPECT_EQ(dim.Get(0), 10);
    EXPECT_EQ(dim.Get(1), 20);
}

// ------------------------------- Setters ---------------------------------------

TEST(Tensor, SetNameUpdatesName) {
    Tensor t;
    t.setName("A");
    EXPECT_EQ(t.name(), "A");
    t.setName("B");
    EXPECT_EQ(t.name(), "B");
}

TEST(Tensor, SetTypeUpdatesType) {
    Tensor t;
    t.setType(data_type::TensorProto_DataType_FLOAT);
    EXPECT_EQ(t.type(), data_type::TensorProto_DataType_FLOAT);
    t.setType(data_type::TensorProto_DataType_DOUBLE);
    EXPECT_EQ(t.type(), data_type::TensorProto_DataType_DOUBLE);
}

TEST(Tensor, SetKindUpdatesKind) {
    Tensor t;
    t.setKind(Tensor_kind::input);
    EXPECT_EQ(t.kind(), Tensor_kind::input);
    t.setKind(Tensor_kind::constant);
    EXPECT_EQ(t.kind(), Tensor_kind::constant);
}

TEST(Tensor, SetDataUpdatesData) {
    Tensor t;
    t.setData("hello");
    EXPECT_EQ(t.data(), "hello");
    t.setData("");
    EXPECT_EQ(t.data(), "");
}

TEST(Tensor, SetShapeUpdatesShape) {
    Tensor t;
    t.setShape({3, 7, 9});
    ASSERT_EQ(t.shape().size(), 3u);
    EXPECT_EQ(t.shape()[0], 3);
    EXPECT_EQ(t.shape()[1], 7);
    EXPECT_EQ(t.shape()[2], 9);

    t.setShape({});
    EXPECT_TRUE(t.shape().empty());
}

TEST(Tensor, SetDimUpdatesDim) {
    Tensor t;

    auto d1 = makeDim({1, 2, 3});
    t.setDim(d1);

    auto dim = t.dim();
    ASSERT_EQ(dim.size(), 3);
    EXPECT_EQ(dim.Get(0), 1);
    EXPECT_EQ(dim.Get(1), 2);
    EXPECT_EQ(dim.Get(2), 3);

    auto d2 = makeDim({});
    t.setDim(d2);
    EXPECT_EQ(t.dim().size(), 0);
}

// ------------------------------ isConstant ------------------------------------

TEST(Tensor, IsConstantTrueOnlyForConstantKind) {
    Tensor t;

    t.setKind(Tensor_kind::unknown);
    EXPECT_FALSE(t.isConstant());

    t.setKind(Tensor_kind::input);
    EXPECT_FALSE(t.isConstant());

    t.setKind(Tensor_kind::output);
    EXPECT_FALSE(t.isConstant());

    t.setKind(Tensor_kind::intermediate);
    EXPECT_FALSE(t.isConstant());

    t.setKind(Tensor_kind::constant);
    EXPECT_TRUE(t.isConstant());
}

TEST(Tensor, IsConstantWorksWithParameterizedConstructor) {
    Tensor t("C", data_type::TensorProto_DataType_FLOAT, {1}, "x", Tensor_kind::constant);
    EXPECT_TRUE(t.isConstant());

    Tensor u("N", data_type::TensorProto_DataType_FLOAT, {1}, "x", Tensor_kind::input);
    EXPECT_FALSE(u.isConstant());
}
