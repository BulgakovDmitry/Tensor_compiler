#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <variant>
#include <vector>

#include "attribute.hpp"

using namespace tensor_compiler;

// ----------------------------- Constructors ------------------------------------

TEST(Attribute, DefaultConstructorInitializesNameEmptyAndDefaultVariant) {
    Attribute a;

    EXPECT_TRUE(a.get_name().empty());

    ASSERT_TRUE(std::holds_alternative<float>(a.get_value()));
    EXPECT_FLOAT_EQ(std::get<float>(a.get_value()), 0.0f);
}

TEST(Attribute, ParameterizedConstructorStoresNameAndFloatValue) {
    Attribute a("alpha", Attribute::AttrValue{1.25f});

    EXPECT_EQ(a.get_name(), "alpha");
    ASSERT_TRUE(std::holds_alternative<float>(a.get_value()));
    EXPECT_FLOAT_EQ(std::get<float>(a.get_value()), 1.25f);
}

TEST(Attribute, ParameterizedConstructorStoresInt64Value) {
    Attribute a("axis", Attribute::AttrValue{int64_t{42}});

    EXPECT_EQ(a.get_name(), "axis");
    ASSERT_TRUE(std::holds_alternative<int64_t>(a.get_value()));
    EXPECT_EQ(std::get<int64_t>(a.get_value()), 42);
}

TEST(Attribute, ParameterizedConstructorStoresStringValue) {
    Attribute a("mode", Attribute::AttrValue{std::string{"nearest"}});

    EXPECT_EQ(a.get_name(), "mode");
    ASSERT_TRUE(std::holds_alternative<std::string>(a.get_value()));
    EXPECT_EQ(std::get<std::string>(a.get_value()), "nearest");
}

TEST(Attribute, ParameterizedConstructorStoresFloatVectorValue) {
    std::vector<float> v = {0.1f, -2.0f, 3.5f};
    Attribute a("scales", Attribute::AttrValue{v});

    EXPECT_EQ(a.get_name(), "scales");
    ASSERT_TRUE(std::holds_alternative<std::vector<float>>(a.get_value()));
    const auto& got = std::get<std::vector<float>>(a.get_value());

    ASSERT_EQ(got.size(), v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        EXPECT_FLOAT_EQ(got[i], v[i]);
    }
}

TEST(Attribute, ParameterizedConstructorStoresInt64VectorValue) {
    std::vector<int64_t> v = {1, 2, 3, 10000000000LL};
    Attribute a("shape", Attribute::AttrValue{v});

    EXPECT_EQ(a.get_name(), "shape");
    ASSERT_TRUE(std::holds_alternative<std::vector<int64_t>>(a.get_value()));
    const auto& got = std::get<std::vector<int64_t>>(a.get_value());

    ASSERT_EQ(got.size(), v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        EXPECT_EQ(got[i], v[i]);
    }
}

// ------------------------------- Getters ---------------------------------------

TEST(Attribute, GetValueReturnsConstReferenceToVariant) {
    Attribute a("beta", Attribute::AttrValue{2.0f});

    // проверяем, что возвращается const ссылка (типово) и значение корректное
    const Attribute::AttrValue& ref = a.get_value();
    ASSERT_TRUE(std::holds_alternative<float>(ref));
    EXPECT_FLOAT_EQ(std::get<float>(ref), 2.0f);
}

// ------------------------------- set_value -------------------------------------

TEST(Attribute, SetValueUpdatesVariantButKeepsName) {
    Attribute a("attr", Attribute::AttrValue{1.0f});
    EXPECT_EQ(a.get_name(), "attr");

    a.set_value(Attribute::AttrValue{int64_t{7}});
    EXPECT_EQ(a.get_name(), "attr");
    ASSERT_TRUE(std::holds_alternative<int64_t>(a.get_value()));
    EXPECT_EQ(std::get<int64_t>(a.get_value()), 7);

    a.set_value(Attribute::AttrValue{std::string{"relu"}});
    EXPECT_EQ(a.get_name(), "attr");
    ASSERT_TRUE(std::holds_alternative<std::string>(a.get_value()));
    EXPECT_EQ(std::get<std::string>(a.get_value()), "relu");
}

TEST(Attribute, SetValueCanSwitchToVectorTypes) {
    Attribute a("v", Attribute::AttrValue{int64_t{0}});

    std::vector<float> vf = {1.0f, 2.0f};
    a.set_value(Attribute::AttrValue{vf});
    ASSERT_TRUE(std::holds_alternative<std::vector<float>>(a.get_value()));
    const auto& gotf = std::get<std::vector<float>>(a.get_value());
    ASSERT_EQ(gotf.size(), 2u);
    EXPECT_FLOAT_EQ(gotf[0], 1.0f);
    EXPECT_FLOAT_EQ(gotf[1], 2.0f);

    std::vector<int64_t> vi = {10, 20, 30};
    a.set_value(Attribute::AttrValue{vi});
    ASSERT_TRUE(std::holds_alternative<std::vector<int64_t>>(a.get_value()));
    const auto& goti = std::get<std::vector<int64_t>>(a.get_value());
    ASSERT_EQ(goti.size(), 3u);
    EXPECT_EQ(goti[0], 10);
    EXPECT_EQ(goti[1], 20);
    EXPECT_EQ(goti[2], 30);
}