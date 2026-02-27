#include <gtest/gtest.h>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "tensor.hpp"

using namespace tensor_compiler;

// ------------------------------ Helpers ----------------------------------------

static std::vector<float> unpack_floats_from_raw(const std::string& raw) {
    EXPECT_EQ(raw.size() % sizeof(float), 0u);

    std::vector<float> out(raw.size() / sizeof(float));
    if (!raw.empty()) {
        std::memcpy(out.data(), raw.data(), raw.size());
    }
    return out;
}

static dim_type make_dim(std::initializer_list<int64_t> dims) {
    dim_type d;
    for (auto v : dims) d.Add(v);
    return d;
}

// ----------------------------- Constructors ------------------------------------

TEST(Tensor, DefaultConstructorHasExpectedDefaults) {
    Tensor t;

    EXPECT_TRUE(t.get_name().empty());
    EXPECT_EQ(t.get_type(), data_type::TensorProto_DataType_UNDEFINED);
    EXPECT_EQ(t.get_kind(), Tensor_kind::unknown);

    EXPECT_TRUE(t.get_data().empty());
    EXPECT_TRUE(t.get_shape().empty());

    auto dim = t.get_dim();
    EXPECT_EQ(dim.size(), 0);
}

TEST(Tensor, ParameterizedConstructorStoresFields) {
    std::string name = "W";
    std::vector<int64_t> shape = {2, 3};
    std::string data = "rawbytes";
    Tensor_kind kind = Tensor_kind::constant;

    Tensor t(name, data_type::TensorProto_DataType_FLOAT, shape, data, kind);

    EXPECT_EQ(t.get_name(), "W");
    EXPECT_EQ(t.get_type(), data_type::TensorProto_DataType_FLOAT);
    EXPECT_EQ(t.get_kind(), Tensor_kind::constant);
    EXPECT_EQ(t.get_data(), "rawbytes");

    ASSERT_EQ(t.get_shape().size(), 2u);
    EXPECT_EQ(t.get_shape()[0], 2);
    EXPECT_EQ(t.get_shape()[1], 3);

    auto dim = t.get_dim();
    EXPECT_EQ(dim.size(), 0);
}

// ------------------------------- create ----------------------------------------

TEST(Tensor, CreateStoresNameShapeKindAndFloatType) {
    std::string name = "X";
    std::vector<int64_t> shape = {1, 4};
    std::vector<float> data = {1.0f, 2.0f, 3.5f, -4.25f};

    Tensor t = Tensor::create(name, shape, data, Tensor_kind::input);

    EXPECT_EQ(t.get_name(), "X");
    EXPECT_EQ(t.get_kind(), Tensor_kind::input);
    EXPECT_EQ(t.get_type(), data_type::TensorProto_DataType_FLOAT);

    ASSERT_EQ(t.get_shape().size(), 2u);
    EXPECT_EQ(t.get_shape()[0], 1);
    EXPECT_EQ(t.get_shape()[1], 4);

    EXPECT_EQ(t.get_data().size(), data.size() * sizeof(float));

    auto unpacked = unpack_floats_from_raw(t.get_data());
    ASSERT_EQ(unpacked.size(), data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        EXPECT_FLOAT_EQ(unpacked[i], data[i]);
    }
}

TEST(Tensor, CreateWithEmptyDataProducesEmptyRawData) {
    Tensor t = Tensor::create("Empty", {2, 2}, {}, Tensor_kind::intermediate);

    EXPECT_EQ(t.get_name(), "Empty");
    EXPECT_EQ(t.get_kind(), Tensor_kind::intermediate);
    EXPECT_EQ(t.get_type(), data_type::TensorProto_DataType_FLOAT);

    ASSERT_EQ(t.get_shape().size(), 2u);
    EXPECT_EQ(t.get_shape()[0], 2);
    EXPECT_EQ(t.get_shape()[1], 2);

    EXPECT_TRUE(t.get_data().empty());
}

// ------------------------------- Getters ---------------------------------------

TEST(Tensor, GettersReturnSetValues) {
    Tensor t;

    t.set_name("T");
    t.set_type(data_type::TensorProto_DataType_INT64);
    t.set_kind(Tensor_kind::output);
    t.set_data("abc");
    t.set_shape({10, 20});
    t.set_dim(make_dim({10, 20}));

    EXPECT_EQ(t.get_name(), "T");
    EXPECT_EQ(t.get_type(), data_type::TensorProto_DataType_INT64);
    EXPECT_EQ(t.get_kind(), Tensor_kind::output);
    EXPECT_EQ(t.get_data(), "abc");

    ASSERT_EQ(t.get_shape().size(), 2u);
    EXPECT_EQ(t.get_shape()[0], 10);
    EXPECT_EQ(t.get_shape()[1], 20);

    auto dim = t.get_dim();
    ASSERT_EQ(dim.size(), 2);
    EXPECT_EQ(dim.Get(0), 10);
    EXPECT_EQ(dim.Get(1), 20);
}

// ------------------------------- Setters ---------------------------------------

TEST(Tensor, SetNameUpdatesName) {
    Tensor t;
    t.set_name("A");
    EXPECT_EQ(t.get_name(), "A");
    t.set_name("B");
    EXPECT_EQ(t.get_name(), "B");
}

TEST(Tensor, SetTypeUpdatesType) {
    Tensor t;
    t.set_type(data_type::TensorProto_DataType_FLOAT);
    EXPECT_EQ(t.get_type(), data_type::TensorProto_DataType_FLOAT);
    t.set_type(data_type::TensorProto_DataType_DOUBLE);
    EXPECT_EQ(t.get_type(), data_type::TensorProto_DataType_DOUBLE);
}

TEST(Tensor, SetKindUpdatesKind) {
    Tensor t;
    t.set_kind(Tensor_kind::input);
    EXPECT_EQ(t.get_kind(), Tensor_kind::input);
    t.set_kind(Tensor_kind::constant);
    EXPECT_EQ(t.get_kind(), Tensor_kind::constant);
}

TEST(Tensor, SetDataUpdatesData) {
    Tensor t;
    t.set_data("hello");
    EXPECT_EQ(t.get_data(), "hello");
    t.set_data("");
    EXPECT_EQ(t.get_data(), "");
}

TEST(Tensor, SetShapeUpdatesShape) {
    Tensor t;
    t.set_shape({3, 7, 9});
    ASSERT_EQ(t.get_shape().size(), 3u);
    EXPECT_EQ(t.get_shape()[0], 3);
    EXPECT_EQ(t.get_shape()[1], 7);
    EXPECT_EQ(t.get_shape()[2], 9);

    t.set_shape({});
    EXPECT_TRUE(t.get_shape().empty());
}

TEST(Tensor, SetDimUpdatesDim) {
    Tensor t;

    auto d1 = make_dim({1, 2, 3});
    t.set_dim(d1);

    auto dim = t.get_dim();
    ASSERT_EQ(dim.size(), 3);
    EXPECT_EQ(dim.Get(0), 1);
    EXPECT_EQ(dim.Get(1), 2);
    EXPECT_EQ(dim.Get(2), 3);

    auto d2 = make_dim({});
    t.set_dim(d2);
    EXPECT_EQ(t.get_dim().size(), 0);
}

// ------------------------------ is_constant ------------------------------------

TEST(Tensor, IsConstantTrueOnlyForConstantKind) {
    Tensor t;

    t.set_kind(Tensor_kind::unknown);
    EXPECT_FALSE(t.is_constant());

    t.set_kind(Tensor_kind::input);
    EXPECT_FALSE(t.is_constant());

    t.set_kind(Tensor_kind::output);
    EXPECT_FALSE(t.is_constant());

    t.set_kind(Tensor_kind::intermediate);
    EXPECT_FALSE(t.is_constant());

    t.set_kind(Tensor_kind::constant);
    EXPECT_TRUE(t.is_constant());
}

TEST(Tensor, IsConstantWorksWithParameterizedConstructor) {
    Tensor t("C", data_type::TensorProto_DataType_FLOAT, {1}, "x", Tensor_kind::constant);
    EXPECT_TRUE(t.is_constant());

    Tensor u("N", data_type::TensorProto_DataType_FLOAT, {1}, "x", Tensor_kind::input);
    EXPECT_FALSE(u.is_constant());
}