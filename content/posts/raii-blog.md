---
title: "llama-cpp 源码学习: C++ RAII 详解"
date: 2026-05-07
tags:
  - raii
  - cpp
  - llama-cpp
description: RAII 是 C++ 中最核心的资源管理惯用法——构造时获取资源，析构时释放资源。本文从 C 语言的痛点出发，讲清 RAII 的机制、所有权语义，并结合 llama.cpp 中的实际应用，展示 unique_ptr + 自定义删除器如何优雅包装 C API。
toc: true
draft: false
---

## 1. 定义

**RAII** = **R**esource **A**cquisition **I**s **I**nitialization（资源获取即初始化）。

**一句话**：构造时获取资源，析构时释放资源。你只管拿，不用管还。

名字有点反直觉——强调「获取」，但真正价值在于析构时的**自动释放**。

RAII 是 C++ 中的一个**编程惯用法（idiom）**，不是语言关键字或语法。它利用 C++ 的两条语言规则：

- 构造函数和析构函数
- 栈上对象离开作用域时析构函数一定被调用（**没有例外**）

---

## 2. 为什么需要 RAII — C 语言的痛点

在 C 中，每获取一个资源，脑中就多一份「什么时候还」的负担：

```c
// ❌ C 风格：多个资源 × 多个退出点 → 组合爆炸
int process(const char * path) {
    FILE * fp = fopen(path, "rb");
    if (!fp) return -1;

    float * buf = malloc(1024 * sizeof(float));
    if (!buf) {
        fclose(fp);           // 别忘了关文件！
        return -2;
    }

    if (fread(buf, sizeof(float), 1024, fp) != 1024) {
        free(buf);            // 别忘了释放！
        fclose(fp);           // 别忘了关闭！
        return -3;
    }

    // 正常路径：同样的 clean up 代码
    free(buf);
    fclose(fp);
    return 0;
}
// 问题：3 个 return 点，每个都要手动清理
// 再加一个资源 → 每个退出点都要加一行 free → 心智负担 O(n×m)
```

更致命的是：**C++ 有异常，而异常会跳过正常 return 路径**。即使你写了 clean up 代码，异常也能让它永远不执行：

```cpp
// ❌ C 风格在 C++ 中：异常会绕过 clean up
void bad() {
    float * buf = new float[1024];
    do_something(buf);  // ← 如果这里抛异常……
    delete[] buf;       // ← 这行永远不会执行！内存泄漏
}
```

**RAII 版**：

```cpp
// ✅ RAII：无论怎么退出，资源自动释放
void good(const std::string & path) {
    std::ifstream fp(path, std::ios::binary);  // 构造：打开文件
    std::vector<float> buf(1024);               // 构造：分配内存

    fp.read(reinterpret_cast<char*>(buf.data()), 1024 * sizeof(float));
    // ... 任意逻辑 ...

    if (出错) return;        // 早期 return → 析构自动执行
    // 即使抛异常 → 栈展开保证析构自动执行
}   // 离开作用域：buf 析构释放内存 → fp 析构关闭文件
```

---

## 3. 机制：三要素

RAII 依赖 C++ 的铁律——**栈上对象离开作用域，析构函数一定被调用**：

```
进入作用域 {         ← 自动变量构造
    资源持有者 obj;
    使用 obj...
    if (...) return;  ← 栈展开，析构照常调用
    可能抛异常...      ← 栈展开，析构照常调用
}                     ← 离开作用域，析构自动执行
```

三要素：

1. **构造** → 获取资源（开文件、分配内存、加锁...）
2. **使用** → 正常操作
3. **析构** → 释放资源（关文件、释放内存、解锁...）

其中第 3 步由编译器自动插入，你不需写任何释放代码。

### 为什么强调「栈上」？

内存分两个主要区域：

| | 栈 (Stack) | 堆 (Heap) |
|--|-----------|----------|
| 怎么创建 | 直接声明变量 `int a;` | `new` / `malloc` |
| 谁管释放 | **编译器自动**（离开作用域就销毁） | **你手动** `delete` / `free` |
| 生命周期 | 确定，出大括号就死 | 不确定，你什么时候 `delete` 什么时候死 |

```cpp
void foo() {
    int a = 42;                          // 栈上对象
    std::vector<float> buf(128);         // 栈上对象（内部数组在堆上，vector 析构时自动释放）

    int * p = new int(42);               // p 本身在栈上，但 *p 在堆上！
    std::unique_ptr<int> q = std::make_unique<int>(42);  // q 在栈上，*q 在堆上

    // 离开函数时：
    //   a  → 自动销毁（栈）
    //   buf → 自动销毁，析构时释放内部堆内存
    //   p  → 栈上的指针本身销毁，但 *p 泄漏了！没 delete
    //   q  → 自动销毁，unique_ptr 析构时帮你 delete 堆上的 int
}
```

RAII 只对**栈上对象**有效，因为只有栈上对象才有「离开作用域必析构」的保证。`unique_ptr` 的精妙之处在于：**自身在栈上，持有堆上对象**——用栈的确定性去管理堆的不确定性。

简单记：栈上 = 自动挡，堆上 = 手动挡。RAII = 给手动挡装个自动离合器。

---

## 4. 四重价值

| 价值              | 说明                                               |
| --------------- | ------------------------------------------------ |
| **自动 clean up** | 彻底解决手动释放问题，一个 return 和十个 return 一样简单             |
| **异常安全**        | 栈展开保证析构执行，这是 C 语言做不到的                            |
| **可组合性**        | 多资源时编译器保证按构造逆序析构，顺序永远正确                          |
| **确定性释放**       | 离开作用域那一刻释放，不像 GC 不可预测；且管理范围不限于内存（文件、锁、GPU 显存...） |

**可组合性示例**：

```cpp
void composite() {
    A a;     // 1. a 构造
    B b(a);  // 2. b 构造（依赖 a）
    C c;     // 3. c 构造
}            // 析构顺序：c → b → a（编译器保证逆序，永远正确）
```

---

## 5. 所有权语义（RAII 的自然副产品）

所有权语义不是 RAII 本身定义的概念，而是**全面使用 RAII 后的自然效果**。

C 语言中所有资源都是 `T*`，从类型看不出任何所有权信息：

```c
char * name();              // 返回值我 free 吗？还是内部 static buffer？
void process(char * buf);   // 这个函数会 free 我的 buf 吗？
// 答案全在注释或文档里——注释过时就是 bug
```

C++ 用**类型本身**表达所有权：

| 类型 | 所有权语义 | 含义 |
|------|-----------|------|
| `T*` / `T&` | 不拥有 | 路人，路过看看，不负责释放 |
| `std::unique_ptr<T>` | 独占 | 我是唯一主人，我负责释放 |
| `std::shared_ptr<T>` | 共享 | 最后一个持有者负责释放 |

**llama.cpp 中的例子**（`common/common.h:846-861`）：

```cpp
struct common_init_result {
    llama_model   * model();      // 裸指针 → 借你看，别释放，归我管
    llama_context * context();    // 裸指针 → 同上

private:
    std::unique_ptr<impl> pimpl;  // unique_ptr → 内部资源我独占，我负责释放
};
```

调用 `result.model()` 拿到 `llama_model*`，从**类型**就知道——不要 delete 它。所有权从注释搬进了类型系统。

**关系总结**：

```
RAII（设计理念）
  │
  └─ 落地：构造 = 获取，析构 = 释放
        │
        └─ 自然效果：所有权语义
              unique_ptr → 一看就是独占
              shared_ptr → 一看就是共享
              T* / T&    → 一看就是不管释放
```

---

## 6. llama.cpp 中的两种包装手段

### 手段一：`unique_ptr` + 自定义删除器（包装 C API）

C API 用 `ggml_init()` 创建、`ggml_free()` 销毁。但 `unique_ptr` 默认调 `delete`，不兼容。通过**自定义删除器**注入 C 的清理函数：

```cpp
// ggml/include/ggml-cpp.h:17-39

struct ggml_context_deleter {
    void operator()(ggml_context * ctx) { ggml_free(ctx); }
};

using ggml_context_ptr = std::unique_ptr<ggml_context, ggml_context_deleter>;
//                                       ↑ 管理的指针类型           ↑ 怎么销毁它
```

使用效果：

```cpp
void do_ggml_work() {
    ggml_context_ptr ctx(ggml_init(params));  // 构造 = ggml_init
    // ... tensor 操作 ...
}   // 析构 → ggml_context_deleter(ctx) → ggml_free(ctx)，零负担
```

自定义删除器本质上是把 C 的清理函数「注册」到了 `unique_ptr` 的模板参数里。

### 手段二：手写 Wrapper 类（管理一组资源）

有时需要管理的不是单个指针，而是一组生命周期相同的资源（model + context + backend）：

```cpp
// common/common.h:846-861

struct common_init_result {
    common_init_result(common_params & params) {
        // 构造：依次加载 model、创建 context、初始化 backend...
        // 内部 pimpl 持有所有子资源
    }

    ~common_init_result() = default;
    // 析构：什么都不用写！
    // pimpl 是 unique_ptr → 自动 delete impl
    //   → impl 的成员析构
    //     → llama_model_ptr 析构 (llama_free_model)
    //     → llama_context_ptr 析构 (llama_free)
    //     → ggml_backend_ptr 析构 (ggml_backend_free)

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};
```

这里发生了**多级联析构**——你只管构造 `common_init_result`，内部所有子资源自动释放。

**手段一 vs 手段二**：
- 手段一：原子资源包装，适合「这个资源应该像指针一样被传递」的场景
- 手段二：复合资源包装，适合「一组资源生命周期完全相同」的场景

---

## 7. 常见误解

### 「RAII 就是 unique_ptr」

不是。RAII 是**理念**，`unique_ptr` 是实现这个理念的**一个工具**。`std::vector`、`std::ifstream`、`std::lock_guard` 都是 RAII 的实现，但都不是 `unique_ptr`。

### 「和 GC 是一回事」

本质区别：

| | RAII | GC（Java/Go/Python） |
|--|------|----------------------|
| 释放时机 | 确定（离开作用域那一刻） | 不确定（GC 决定何时跑） |
| 管理范围 | 任意资源（内存/文件/锁/GPU 显存） | 只管内存 |
| 性能 | 零运行时开销，时机可预测 | 有 stop-the-world 暂停 |

### 「用了 RAII 就不会泄漏」

循环引用依然需要 `std::weak_ptr` 手动打破。RAII 让泄漏变难了，但不是银弹。

---

## 8. 小结

RAII 的核心思想极其简单——**把资源的生命周期绑到对象上，让编译器替你 clean up**。但这简单理念带来了深远影响：自动资源管理、异常安全、类型级别的所有权表达。在 llama.cpp 中，你看到的每一个 `_ptr` 类型别名和每一个带析构函数的 struct，背后都是 RAII。
