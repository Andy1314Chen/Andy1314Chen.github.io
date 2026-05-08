---
title: "llama-cpp 源码学习: C++ 模板详解"
date: 2026-05-08
tags:
  - templates
  - cpp
  - llama-cpp
description: 模板是 C++ 泛型编程的核心——一份逻辑，多种类型，编译器按需生成代码。本文从宏的痛点出发，讲清函数模板、类模板、全特化、编译期类型计算，并结合 llama.cpp 中的实际用法（类型映射表、非类型参数、static_assert）给出实用规则。
toc: true
draft: false
---

## 1. 为什么需要模板

**核心问题**：一份逻辑，多种类型——不想为每种类型写一遍。

### 1.1 C 的做法：宏

```c
#define MAX(a, b) ((a) > (b) ? (a) : (b))

int    x = MAX(3, 5);       // 对 int 用
float  y = MAX(3.0f, 5.0f); // 对 float 用
```

宏是纯文本替换——没有类型检查、括号地狱、调试困难。

### 1.2 C++ 的做法：模板

```cpp
template<class T>
T max(T a, T b) {
    return a > b ? a : b;
}

int   x = max(3, 5);        // 编译器自动生成 max<int>
float y = max(3.0f, 5.0f);  // 编译器自动生成 max<float>
```

编译器遇到 `max(3, 5)` 时，自动生成一份 `T = int` 的版本——等价于你手写了 `int max(int a, int b)`。这个过程叫**模板实例化**，发生在编译期，零运行时开销。

**类比**：模板是"函数制作机"——你给配方，编译器按需生产具体类型的版本。

---

## 2. 函数模板

### 2.1 基础写法

```cpp
template<class T>     // T 是类型参数，相当于"占位符"
void swap(T & a, T & b) {
    T tmp = a;
    a = b;
    b = tmp;
}
```

`T` 可以是 `int`、`float`、`std::string`……任何支持拷贝和赋值的类型。

### 2.2 为什么有时要特化

通用版本对 `int` / `float` 工作良好：

```cpp
// common/common.h:732-745 — 通用版本，用 stringstream 解析
template<class T>
static std::vector<T> string_split(const std::string & str, char delim) {
    static_assert(!std::is_same<T, std::string>::value,
                  "Please use the specialized version for std::string");
    std::vector<T> values;
    std::istringstream str_stream(str);
    std::string token;
    while (std::getline(str_stream, token, delim)) {
        T value;
        std::istringstream token_stream(token);
        token_stream >> value;
        values.push_back(value);
    }
    return values;
}
```

但 `T = std::string` 不能走 `token_stream >> value`——含空格就截断了。所以需要**特化**：

```cpp
// common/common.h:747-761 — std::string 特化版本
template<>
inline std::vector<std::string> string_split<std::string>(
    const std::string & str, char delim)
{
    std::vector<std::string> parts;
    size_t begin_pos = 0;
    size_t delim_pos = str.find(delim);
    while (delim_pos != std::string::npos) {
        std::string part = str.substr(begin_pos, delim_pos - begin_pos);
        parts.emplace_back(part);
        begin_pos = delim_pos + 1;
        delim_pos = str.find(delim, begin_pos);
    }
    parts.emplace_back(str.substr(begin_pos));
    return parts;
}
```

`template<>` 告诉编译器："`T = std::string` 时，别用通用版，用这份"。这就是**全特化（full specialization）**。

---

## 3. 类模板

你已经在用——`std::vector<int>`、`std::unique_ptr<ggml_context>` 都是类模板。

```cpp
template<class T>
class Box {
    T value;
public:
    Box(T v) : value(v) {}
    T get() { return value; }
};

Box<int>    bi(42);      // value 是 int
Box<std::string> bs("hello");  // value 是 std::string
```

你看过的 `unique_ptr<T>` / `shared_ptr<T>` / `vector<T>` 本质都是类模板——用同一个类定义适配无穷多种类型。

---

## 4. 编译期类型计算 —— 模板的进阶用法

模板最强的能力是**编译期计算**——在编译阶段就把类型关系算好，而不是运行时判断。

### 4.1 类型映射表

llama.cpp 在读写 GGUF 文件时，需要从 C++ 类型（`uint8_t`、`float`……）对应到 GGUF 的枚举值（`GGUF_TYPE_UINT8`、`GGUF_TYPE_FLOAT32`……）。这用模板特化做成了一张"编译期查表"：

```cpp
// ggml/src/gguf.cpp:29-69（节选）
template <typename T>
struct type_to_gguf_type;                      // 主模板：只声明，不定义

template <> struct type_to_gguf_type<uint8_t>  { static constexpr enum gguf_type value = GGUF_TYPE_UINT8; };
template <> struct type_to_gguf_type<int32_t>  { static constexpr enum gguf_type value = GGUF_TYPE_INT32; };
template <> struct type_to_gguf_type<float>    { static constexpr enum gguf_type value = GGUF_TYPE_FLOAT32; };
template <> struct type_to_gguf_type<bool>     { static constexpr enum gguf_type value = GGUF_TYPE_BOOL; };
// ... 共 10 个类型的特化
```

使用时：

```cpp
// ggml/src/gguf.cpp:142
: key(key), is_array(false), type(type_to_gguf_type<T>::value) {}
```

编译器在编译期就把 `type_to_gguf_type<float>::value` 替换为 `GGUF_TYPE_FLOAT32`——**零运行时查找、零内存占用**。

### 4.2 非类型模板参数

模板参数不一定是类型，也可以是**具体的值**（`true`、`42`、`3`）。普通模板参数是类型（`int`、`float`、`std::string`），非类型参数就是一个编译期已知的常量。

```cpp
// 类型参数：T 可以是 int、float、string...
template<class T>
void f(T a) { ... }

// 非类型参数：N 就是一个具体的数字，比如 5、10
template<int N>
void f() {
    int arr[N];  // 编译期就知道数组大小
}
```

本质：模板是在编译期"生成代码"的。类型参数让编译器根据不同**类型**生成不同版本；非类型参数让编译器根据不同**值**生成不同版本。两者都是"一份配方，编译期按需生产"。

llama.cpp 中的例子：

```cpp
// src/models/models.h:119-122
struct llama_model_llama : public llama_model_base {
    template <bool embed>                       // 非类型参数：bool
    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(
        const llm_graph_params & params) const override;
};
```

`template<bool embed>` 允许编译时生成两个图版本：

```cpp
template <> struct llama_model_llama::graph<true>  { /* embedding 模式的图 */ };
template <> struct llama_model_llama::graph<false> { /* 生成模式的图 */ };
```

同一个结构体，`embed=true` 和 `embed=false` 是**完全不同的两个类型**——编译器分别实例化。适合"编译期已知的分支"——不用运行时 `if`，编译阶段就决定用哪份代码。

### 4.3 `static_assert` —— 编译期报错

模板的配套工具：在实例化时就检查条件，不满足直接停止编译。

```cpp
// common/common.h:734
static_assert(!std::is_same<T, std::string>::value,
              "Please use the specialized version for std::string");
```

如果谁写了 `string_split<std::string>(...)` 却不走特化版本，编译直接挂——比运行时 `assert` 早得多，且信息明确。

---

## 5. 模板 vs 宏

|      | 宏          | 模板            |
| ---- | ---------- | ------------- |
| 工作阶段 | 预处理器（文本替换） | 编译器（类型感知）     |
| 类型安全 | 无——什么都往里塞  | 有——类型不匹配编译报错  |
| 调试体验 | 地狱——替换后的乱码 | 正常——编译器知道这是模板 |
| 作用域  | 无视作用域      | 遵守命名空间 / 类作用域 |

模板是 C++ 的答案："我们受了宏 20 年的苦，发明了类型安全、作用域正常的替代品。"

---

## 6. 三条规则

1. **优先用函数重载，重载不够才上模板** — 模板会增加编译时间和错误信息复杂度。能用重载解决的（比如不同参数类型），不一定非要模板。
2. **特化是最后手段** — 先考虑：这个类型的通用版真的不能工作吗？能不改模板定义就不改。
3. **编译错误太长时看第一条报错** — 模板的错误信息从几十行到几百行，只看第一个 `error:` 通常就够定位问题。

---

## 7. 速查表

| 语法                                           | 含义                                     |
| -------------------------------------------- | -------------------------------------- |
| `template<class T>` / `template<typename T>` | 声明类型参数 T（两者等价）                         |
| `template<int N>` / `template<bool B>`       | 非类型模板参数                                |
| `template<>`                                 | 全特化——为某个具体类型/值提供专属实现                   |
| `T::value`                                   | 访问模板参数的静态成员（常用于编译期计算）                  |
| `static_assert(cond, msg)`                   | 编译期条件检查                                |
| `typename T::type`                           | 告诉编译器 `T::type` 是类型而非值（`typename` 消歧义） |
