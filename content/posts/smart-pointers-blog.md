---
title: "llama-cpp 源码学习: C++ 智能指针详解"
date: 2026-05-07
tags:
  - smart-pointers
  - cpp
  - llama-cpp
description: 智能指针是 RAII 理念最典型的工具。本文详解 unique_ptr、自定义删除器、shared_ptr/weak_ptr，并结合 llama.cpp 中的实际用法，给出初学者的三条实用规则。
toc: true
draft: false
---

## 1. 为什么需要智能指针

C++ 中 `new` 出来的堆对象需要 `delete` 释放。一旦中间提前 `return` 或抛出异常，`delete` 没执行到，就内存泄漏了：

```cpp
void bad() {
    int * p = new int(42);
    if (some_condition())
        return;          // 忘了 delete p —— 泄漏了
    do_stuff(p);
    delete p;
}
```

智能指针是一个**栈上对象**，在析构时自动帮你 `delete`：

```cpp
void good() {
    auto p = std::make_unique<int>(42);
    if (some_condition())
        return;          // p 离开作用域，自动 delete —— 不漏
    do_stuff(p.get());
}
```

智能指针是 RAII 理念的最典型工具。

---

## 2. `unique_ptr` — 默认选它

### 2.1 一句话

**独占所有权**：一个 `unique_ptr` 是它所指堆对象的唯一主人。离开作用域时自动释放。

### 2.2 创建

只有一种推荐写法：

```cpp
auto p = std::make_unique<MyClass>(arg1, arg2);
```

`make_unique` 直接在堆上构造对象并返回一个 `unique_ptr`。llama.cpp 中有数百处 `make_unique` 使用，几乎见不到裸 `new`。

### 2.3 使用

和裸指针一样用——解引用、箭头、判空：

```cpp
auto p = std::make_unique<std::string>("hello");

*p = "world";           // 解引用
p->size();              // 成员访问
if (p) { /* ... */ }   // 非空时为 true
```

### 2.4 不能拷贝，只能移动

`unique_ptr` 不能拷贝（否则哪有唯一主人？），但可以**转移所有权**：

```cpp
auto a = std::make_unique<int>(1);
// auto b = a;          // 编译错误：不能拷贝

auto b = std::move(a);  // 所有权从 a 移给 b
// a 现在是空的，不要再用了
```

在 llama.cpp 中你经常看到这种转移：

```cpp
// src/llama-model.cpp:1501
pimpl->ctxs_bufs.emplace_back(std::move(ctx_ptr), std::move(bufs));
```

`ctx_ptr` 和 `bufs` 的所有权被移入容器，之后原变量不再持有资源。

### 2.5 llama.cpp 实例

```cpp
// include/llama-cpp.h:27-29
typedef std::unique_ptr<llama_model,   llama_model_deleter>   llama_model_ptr;
typedef std::unique_ptr<llama_context, llama_context_deleter> llama_context_ptr;
typedef std::unique_ptr<llama_sampler, llama_sampler_deleter> llama_sampler_ptr;
```

这些是 llama.cpp 对 C API 资源的标准包装：用 `unique_ptr` 保证模型、上下文、采样器不会泄漏。

```cpp
// common/common.h:863
using common_init_result_ptr = std::unique_ptr<common_init_result>;
```

这里没指定删除器——因为 `common_init_result` 是 C++ 类，析构函数已经挂好了资源释放链，`unique_ptr` 默认 `delete` 就够了。

---

## 3. 自定义删除器 — 给 C API 装 RAII

### 3.1 问题

`unique_ptr` 默认用 `delete` 释放资源。但 llama.cpp 底层大量 C API，释放函数叫 `ggml_free`、`llama_model_free`，不是 `delete`。

### 3.2 解法：写一个「删除器」struct

考虑一个玩具例子——管理文件句柄：

```cpp
struct FileCloser {
    void operator()(FILE * f) { fclose(f); }
};

std::unique_ptr<FILE, FileCloser> f(fopen("data.txt", "r"));
// f 离开作用域时，自动调用 FileCloser::operator() → fclose
```

`unique_ptr` 的第二个模板参数就是删除器类型。只要这个类型有 `operator()(T*)`，`unique_ptr` 析构时就会调它。

### 3.3 llama.cpp 中的真实写法

```cpp
// ggml/include/ggml-cpp.h:17,20
struct ggml_context_deleter {
    void operator()(ggml_context * ctx) { ggml_free(ctx); }
};

typedef std::unique_ptr<ggml_context, ggml_context_deleter> ggml_context_ptr;
```

`ggml_context` 是 ggml 的计算上下文，必须用 `ggml_free` 释放。`ggml_context_deleter` 就是为这个 C API 定制的删除器。

项目中 **20+ 种**删除器都是这个模式：

```cpp
struct xxx_deleter { void operator()(xxx * p) { xxx_free(p); } };
```

看懂一个，全看懂了。

---

## 4. `shared_ptr` 和 `weak_ptr` — 看得懂就行

### 4.1 重要前提

**llama.cpp 核心层（`src/llama-*`、`common/`）几乎不用 `shared_ptr`。** 主要在 Vulkan、WebGPU、OpenVINO 等后端子目录大量使用。初学阶段默认用 `unique_ptr`，遇到 `shared_ptr` 能看懂即可。

### 4.2 `shared_ptr` — 共享所有权

多个 `shared_ptr` 指向同一份数据，**引用计数**记录有多少个主人。最后一个归零时释放。

```cpp
auto a = std::make_shared<int>(42);
auto b = a;          // 引用计数 = 2
// a 和 b 都离开作用域后，计数归零，释放内存
```

llama.cpp 中罕见的共享场景：

```cpp
// src/llama-batch.h:68
std::shared_ptr<data_t> data;
```

多个 `llama_ubatch` 可能共享同一份底层数据，一份数据被多个视角引用，所以用 `shared_ptr`。

### 4.3 `weak_ptr` — 不拥有，只观察

不参与引用计数，不延长生命周期。核心用途：**打破循环引用**。

```cpp
auto a = std::make_shared<int>(42);
std::weak_ptr<int> w = a;   // 不增加引用计数

if (auto sp = w.lock()) {   // lock 提升为 shared_ptr
    *sp = 100;              // 安全使用——a 还没被释放
}
// a 离开作用域后，w.lock() 返回空
```

llama.cpp 核心层 **0 处** `weak_ptr`，仅在 Vulkan、RPC 后端用于打破循环引用和缓存回收。

### 4.4 初学者原则

**默认用 `unique_ptr`，不要急着用 `shared_ptr`。** 确认有多处需要共享同一份数据时，才考虑 `shared_ptr`。

---

## 5. 初学者的三条规则

### 规则 1：默认用 `unique_ptr`

绝大多数场景只有一个明确主人。先 `unique_ptr`，确认需要共享时才换 `shared_ptr`。

```cpp
// 好：明确唯一主人
auto engine = std::make_unique<Engine>();

// 差：没理由就 shared
auto engine = std::make_shared<Engine>();
```

### 规则 2：用 `make_unique` / `make_shared` 创建，不要直接 `new`

```cpp
// 差
std::unique_ptr<Foo> p(new Foo(1, 2, 3));

// 好
auto p = std::make_unique<Foo>(1, 2, 3);
```

两个原因：(1) 少打一次类型名；(2) 异常安全——极端情况下 `new` 和 `unique_ptr` 构造之间如果抛异常，可能泄漏。`make_unique` 没有这个问题。

### 规则 3：别把同一个裸指针传给两个智能指针

```cpp
auto raw = new int(42);

auto a = std::unique_ptr<int>(raw);
auto b = std::unique_ptr<int>(raw);  // 两个都以为自己是主人
// a 和 b 析构时都会 delete raw —— double free！
```

**永远不要让裸指针直接构造智能指针**。用 `make_unique` / `make_shared` 从源头避免。

---

## 6. llama.cpp 智能指针速查表

| 类型别名 | 文件:行号 | 管理什么 |
|----------|-----------|----------|
| `llama_model_ptr`         | `include/llama-cpp.h:27`        | 模型 |
| `llama_context_ptr`       | `include/llama-cpp.h:28`        | 推理上下文 |
| `llama_sampler_ptr`       | `include/llama-cpp.h:29`        | 采样器链 |
| `ggml_context_ptr`        | `ggml/include/ggml-cpp.h:20`    | ggml 计算上下文 |
| `ggml_backend_ptr`        | `ggml/include/ggml-cpp.h:36`    | 后端实例（CPU/CUDA/Metal） |
| `ggml_backend_buffer_ptr` | `ggml/include/ggml-cpp.h:37`    | 后端缓冲区 |
| `ggml_backend_sched_ptr`  | `ggml/include/ggml-cpp.h:39`    | 后端调度器 |
| `common_init_result_ptr`  | `common/common.h:863`           | 推理初始化结果（PImpl） |

---

## 7. 总结

| 你学到了 | 一句话 |
|----------|--------|
| `unique_ptr` 是什么 | 独占所有权的栈上管家，离开作用域自动释放 |
| 自定义删除器怎么写 | `struct Deleter { void operator()(T * p) { free_func(p); } };` |
| `shared_ptr` 是什么 | 引用计数共享所有权 |
| `weak_ptr` 是什么 | 不拥有，只观察，破循环引用 |
| 三条规则 | 默认 unique；用 `make_xxx` 创建；不传裸指针给两个智能指针 |
