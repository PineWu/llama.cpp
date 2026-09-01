---
name: lpu-dev-workflow
description: LPU 后端开发工作流：运行测试、提交代码、推送到 pinewu 远程仓库。Use when the user wants to test LPU backend, commit LPU changes, push to pinewu GitHub, or run the full LPU dev cycle (test → commit → push).
---

LPU 后端开发的三段工作流：**test → commit → push**。每段独立可运行，也可串联执行。

仓库根目录：`/home/lixiang/Workspace/knowledge/llm_infra/llama.cpp`

---

## 1. 运行 LPU 后端测试

**前提**：若 stub 库不存在，先建：
```bash
mkdir -p /tmp/lpu_stub/lib /tmp/lpu_stub/include
cc -shared -o /tmp/lpu_stub/lib/liblpu_runtime.so -x c /dev/null
cc -shared -o /tmp/lpu_stub/lib/liblpu_nn.so      -x c /dev/null
```

**构建**：
```bash
cd /home/lixiang/Workspace/knowledge/llm_infra/llama.cpp
cmake -B build -DGGML_LPU=ON -DLPU_INSTALL_DIR=/tmp/lpu_stub -DCMAKE_BUILD_TYPE=Release
cmake --build build --target test-lpu-backend -j$(nproc)
```

**运行**：
```bash
LD_LIBRARY_PATH=/tmp/lpu_stub/lib ./build/bin/test-lpu-backend
```

**完成标准**：输出 `0 failed`，exit code 0。任何 `FAIL` 行或非零 exit code 须先修复，再进入下一段。

---

## 2. 提交本地代码修改

只暂存 LPU 相关文件，不要用 `git add -A`。

**已跟踪但修改的文件**（按需选取）：
- `CLAUDE.md`
- `ggml/CMakeLists.txt`
- `ggml/src/CMakeLists.txt`
- `ggml/src/ggml-backend-reg.cpp`
- `tests/CMakeLists.txt`

**LPU 源码目录**（新文件一起 add）：
- `ggml/include/ggml-lpu.h`
- `ggml/src/ggml-lpu/`（整个目录）
- `tests/test-lpu-backend.cpp`

**LPU 文档**（新文件）：
- `docs/lpu-*.md`
- `docs/lpu-ffn-internal-ops.md`

**不要提交**：`.understand-anything/`、`docs/architecture-analysis.md`、`docs/backend-architecture.md`、`docs/moe-analysis.md`、`docs/op-scheduling-analysis.md`

```bash
git add <具体文件列表>
git status   # 确认暂存区只包含 LPU 相关文件
```

**Commit message 格式**（中文，HEREDOC 传入）：
```bash
git commit -m "$(cat <<'EOF'
新增 LPU ggml 后端实现：<本次变更摘要>

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

**完成标准**：`git status` 显示工作区干净，`git log --oneline -1` 可见新 commit。

---

## 3. Push 到 pinewu 远程仓库

**确认 remote**：
```bash
git remote -v   # 确认 pinewu 指向 git@github.com:PineWu/llama.cpp.git
```

若 remote 不存在或 URL 是 HTTPS，先修正为 SSH（HTTPS 在非交互式终端下无法认证）：
```bash
git remote add pinewu git@github.com:PineWu/llama.cpp.git
# 或
git remote set-url pinewu git@github.com:PineWu/llama.cpp.git
```

**普通推送**（新 commit）：
```bash
git push pinewu HEAD:lpu-dev
```

**若本次是 amend**（改写了已推送的 commit），需 force push：
```bash
git push --force pinewu HEAD:lpu-dev
```

**完成标准**：push 命令输出 `lpu-dev -> lpu-dev`，无 error 行。若报权限错误，检查本机 SSH key 是否已添加到 GitHub 账号 PineWu。
