from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import textwrap
from pathlib import Path
from typing import Any


SUPPORTED_LANGUAGES = {"python", "cpp", "java", "csharp", "c", "go"}


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.rstrip() for ln in text.split("\n")]
    return "\n".join(lines).strip()


def normalize_output_tokens(text: str) -> list[str]:
    return normalize_text(text).split()


def _try_float(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


def outputs_match(stdout: str, expected: str) -> bool:
    got_norm = normalize_text(stdout)
    exp_norm = normalize_text(expected)
    if got_norm == exp_norm:
        return True
    got_tokens = normalize_output_tokens(stdout)
    exp_tokens = normalize_output_tokens(expected)
    if got_tokens == exp_tokens:
        return True
    if got_tokens and len(got_tokens) == len(exp_tokens):
        numeric_ok = True
        for got, exp in zip(got_tokens, exp_tokens):
            got_float = _try_float(got)
            exp_float = _try_float(exp)
            if got_float is None or exp_float is None:
                numeric_ok = False
                break
            tolerance = max(1e-9, 1e-7 * max(abs(got_float), abs(exp_float), 1.0))
            if abs(got_float - exp_float) > tolerance:
                numeric_ok = False
                break
        if numeric_ok:
            return True
    bool_words = {"yes", "no", "true", "false"}
    if got_tokens and exp_tokens:
        got_lower = [x.lower() for x in got_tokens]
        exp_lower = [x.lower() for x in exp_tokens]
        if got_lower == exp_lower and set(got_lower).issubset(bool_words):
            return True
    return False


def decode_escaped_for_prompt(text: str) -> str:
    if "\n" in text or "\t" in text:
        return text
    return text.replace("\\n", "\n").replace("\\t", "\t")


def parse_fim_prompt(fim_prompt: str) -> tuple[str, str]:
    if "<|fim_prefix|>" not in fim_prompt or "<|fim_suffix|>" not in fim_prompt or "<|fim_middle|>" not in fim_prompt:
        return "", ""
    pre = fim_prompt.split("<|fim_suffix|>", 1)[0].replace("<|fim_prefix|>", "", 1)
    rest = fim_prompt.split("<|fim_suffix|>", 1)[1]
    suf = rest.split("<|fim_middle|>", 1)[0]
    return pre, suf


def sanitize_prediction(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # FIX1: 截断所有可能的 special token，防止泄漏进生成代码
    for stop_token in ["<|im_end|>", "<|endoftext|>", "<|im_start|>",
                       "<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>"]:
        if stop_token in text:
            text = text.split(stop_token, 1)[0]
    if "```" in text:
        text = text.replace("```python", "").replace("```", "")
    return text.strip("\n")


def extract_python_function_name(source: str) -> str | None:
    m = re.search(r"^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", source, flags=re.MULTILINE)
    return m.group(1) if m else None


def toolchain_available(language: str) -> tuple[bool, str]:
    if language == "python":
        return True, ""
    if language == "cpp":
        return (shutil.which("g++") is not None, "g++ not found")
    if language == "c":
        return (shutil.which("gcc") is not None, "gcc not found")
    if language == "java":
        ok = shutil.which("javac") is not None and shutil.which("java") is not None
        return (ok, "javac/java not found")
    if language == "go":
        return (shutil.which("go") is not None, "go not found")
    if language == "csharp":
        ok = (shutil.which("csc") is not None or shutil.which("mcs") is not None) and shutil.which("mono") is not None
        return (ok, "csc-or-mcs/mono not found")
    return False, f"unsupported language {language}"


def run_cmd(
    cmd: list[str],
    cwd: str | None = None,
    timeout_sec: int = 10,
    stdin_text: str | None = None,
) -> tuple[int, str, str, bool]:
    input_bytes = stdin_text.encode("utf-8") if stdin_text is not None else None
    try:
        proc = subprocess.run(
            cmd,
            input=input_bytes,
            capture_output=True,
            cwd=cwd,
            timeout=timeout_sec,
        )
        stdout = (proc.stdout or b"").decode("utf-8", errors="replace")
        stderr = (proc.stderr or b"").decode("utf-8", errors="replace")
        return proc.returncode, stdout, stderr, False
    except subprocess.TimeoutExpired as exc:
        stdout_bytes = exc.stdout or b""
        stderr_bytes = exc.stderr or b""
        if isinstance(stdout_bytes, str):
            stdout = stdout_bytes
        else:
            stdout = stdout_bytes.decode("utf-8", errors="replace")
        if isinstance(stderr_bytes, str):
            stderr = stderr_bytes
        else:
            stderr = stderr_bytes.decode("utf-8", errors="replace")
        return -1, stdout, stderr, True


def judge_humaneval_python(full_code: str, row: dict[str, Any], timeout_sec: int) -> tuple[bool | None, str, str]:
    test_src = str(row.get("judge_payload", {}).get("test", ""))
    if not test_src.strip():
        return None, "unsupported", "missing humaneval test"

    entry_point = str(row.get("metadata", {}).get("entry_point", "")).strip()
    if not entry_point:
        entry_point = extract_python_function_name(full_code) or "candidate"

    py_imports = "from typing import List, Dict, Tuple, Optional, Set, Any\nimport math\nimport collections\n\n"
    program = f"{py_imports}{full_code}\n\n{test_src}\n\ncheck({entry_point})\n"
    with tempfile.TemporaryDirectory() as td:
        py_path = Path(td) / "main.py"
        py_path.write_text(program, encoding="utf-8")
        rc, stdout, stderr, timed_out = run_cmd(["python", str(py_path)], timeout_sec=timeout_sec)
        if timed_out:
            return False, "timeout", stderr[:300]
        if rc == 0:
            return True, "ok", ""
        return False, "runtime_error", (stderr or stdout)[:300]


def extract_check_arg(test_src: str) -> str:
    """从 'def check(fn_name):' 里提取函数名"""
    m = re.search(r"def\s+check\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)", test_src)
    return m.group(1) if m else ""


def python_test_has_check_def(test_src: str) -> bool:
    return re.search(r"(?m)^\s*def\s+check\s*\(", test_src) is not None


def python_test_calls_check(test_src: str) -> bool:
    return re.search(r"(?m)^\s*check\s*\(", test_src) is not None


def python_test_function_calls(test_src: str) -> str:
    if re.search(r"\bunittest\.main\s*\(", test_src):
        return ""
    names = re.findall(r"(?m)^def\s+(test_[A-Za-z_][A-Za-z0-9_]*)\s*\(", test_src)
    seen: set[str] = set()
    calls: list[str] = []
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        if re.search(rf"(?m)^{re.escape(name)}\s*\(", test_src):
            continue
        calls.append(f"{name}()")
    return "\n".join(calls)


def mceval_java_class_name(test_src: str) -> str:
    if re.search(r"\bnew\s+Solution\s*\(", test_src) or re.search(r"\bSolution\s*\.", test_src):
        return "Solution"
    return "Problem"


def mceval_java_helper_fields(full_code: str) -> str:
    helpers: list[str] = []
    if ("dx[" in full_code or "dy[" in full_code) and not re.search(r"\b(?:int|Integer)\s*\[\]\s*dx\b", full_code):
        helpers.append("static int[] dx = {-1, -1, -1, 0, 0, 1, 1, 1};")
        helpers.append("static int[] dy = {-1, 0, 1, -1, 1, -1, 0, 1};")
    if re.search(r"\btree\s*=", full_code) and not re.search(r"\b(?:List|ArrayList)\s*<.*>\s*tree\b", full_code):
        helpers.append("static List<Set<Integer>> tree;")
    if re.search(r"\bres\s*=", full_code) and not re.search(r"\bint\s*\[\]\s*res\b", full_code):
        helpers.append("static int[] res;")
    return "\n".join(helpers)


def prepare_mceval_csharp_test(test_src: str) -> str:
    test_src = re.sub(
        r"Debug\.Assert\s*\((.*)\)\s*;",
        'if (!(\\1)) throw new Exception("assert failed");',
        test_src,
    )
    return test_src


def collapse_duplicate_c_like_signature(full_code: str) -> str:
    return re.sub(
        r"(?m)^([A-Za-z_][A-Za-z0-9_\s\*]+\(.*?\))\s*\n\s*\1\s*\{",
        r"\1 {",
        full_code,
    )


def escape_raw_newlines_in_c_like_literals(source: str) -> str:
    out: list[str] = []
    quote = ""
    escaped = False
    line_comment = False
    block_comment = False
    line_comment_quote = False
    line_comment_escaped = False
    i = 0
    while i < len(source):
        ch = source[i]
        nxt = source[i + 1] if i + 1 < len(source) else ""
        if quote:
            if escaped:
                out.append(ch)
                escaped = False
                i += 1
                continue
            if ch == "\\":
                out.append(ch)
                escaped = True
                i += 1
                continue
            if ch == quote:
                out.append(ch)
                quote = ""
                i += 1
                continue
            if ch == "\n":
                out.append("\\n")
                i += 1
                continue
            if ch == "\r":
                i += 1
                continue
            out.append(ch)
            i += 1
            continue
        if line_comment:
            out.append(ch)
            if ch == "\n":
                line_comment = False
            i += 1
            continue
        if block_comment:
            out.append(ch)
            if ch == "*" and nxt == "/":
                out.append(nxt)
                block_comment = False
                i += 2
            else:
                i += 1
            continue
        if ch == "/" and nxt == "/":
            out.append(ch)
            out.append(nxt)
            line_comment = True
            i += 2
            continue
        if ch == "/" and nxt == "*":
            out.append(ch)
            out.append(nxt)
            block_comment = True
            i += 2
            continue
        if ch in {"'", '"'}:
            quote = ch
        out.append(ch)
        i += 1
    return "".join(out)


def prepare_c_like_source(source: str) -> str:
    out: list[str] = []
    quote = ""
    escaped = False
    line_comment = False
    block_comment = False
    i = 0
    while i < len(source):
        ch = source[i]
        nxt = source[i + 1] if i + 1 < len(source) else ""
        if quote:
            if escaped:
                out.append(ch)
                escaped = False
                i += 1
                continue
            if ch == "\\":
                out.append(ch)
                escaped = True
                i += 1
                continue
            if ch == quote:
                out.append(ch)
                quote = ""
                i += 1
                continue
            if ch == "\n":
                out.append("\\n")
                i += 1
                continue
            if ch == "\r":
                i += 1
                continue
            out.append(ch)
            i += 1
            continue
        if line_comment:
            if line_comment_quote:
                if line_comment_escaped:
                    out.append(ch)
                    line_comment_escaped = False
                    i += 1
                    continue
                if ch == "\\":
                    out.append(ch)
                    line_comment_escaped = True
                    i += 1
                    continue
                if ch == '"':
                    out.append(ch)
                    line_comment_quote = False
                    i += 1
                    continue
                if ch == "\n":
                    out.append("\\n")
                    i += 1
                    continue
                if ch == "\r":
                    i += 1
                    continue
                if ch == "\\" and nxt == "n":
                    out.append("\\n")
                    i += 2
                    continue
                if ch == "\\" and nxt == "r":
                    i += 2
                    continue
                out.append(ch)
                i += 1
                continue
            if ch == '"':
                out.append(ch)
                line_comment_quote = True
                i += 1
                continue
            if ch == "\n":
                out.append(ch)
                line_comment = False
                line_comment_quote = False
                line_comment_escaped = False
                i += 1
                continue
            if ch == "\r":
                i += 1
                continue
            if ch == "\\" and nxt == "n":
                out.append("\n")
                line_comment = False
                line_comment_quote = False
                line_comment_escaped = False
                i += 2
                continue
            if ch == "\\" and nxt == "r":
                i += 2
                continue
            out.append(ch)
            i += 1
            continue
        if block_comment:
            if ch == "\r":
                i += 1
                continue
            if ch == "\\" and nxt == "n":
                out.append("\n")
                i += 2
                continue
            if ch == "\\" and nxt == "r":
                i += 2
                continue
            if ch == "*" and nxt == "/":
                out.append(ch)
                out.append(nxt)
                block_comment = False
                i += 2
                continue
            out.append(ch)
            i += 1
            continue
        if ch == "/" and nxt == "/":
            out.append(ch)
            out.append(nxt)
            line_comment = True
            line_comment_quote = False
            line_comment_escaped = False
            i += 2
            continue
        if ch == "/" and nxt == "*":
            out.append(ch)
            out.append(nxt)
            block_comment = True
            i += 2
            continue
        if ch == "\\" and nxt == "n":
            out.append("\n")
            i += 2
            continue
        if ch == "\\" and nxt == "r":
            i += 2
            continue
        if ch == "\\" and nxt == "t":
            out.append("\t")
            i += 2
            continue
        if ch in {"'", '"'}:
            quote = ch
        out.append(ch)
        i += 1
    return "".join(out)


def wrap_mceval_code(language: str, full_code: str, test_src: str, fn_name_hint: str = "") -> str:
    if language == "python":
        fn_name = (
            fn_name_hint
            or extract_python_function_name(full_code)
            or extract_check_arg(test_src)
            or "candidate"
        )
        py_imports = "\n".join(
            [
                "from typing import *",
                "from math import *",
                "from itertools import *",
                "from functools import *",
                "from collections import *",
                "import bisect",
                "import heapq",
                "import itertools",
                "import functools",
                "import math",
                "import collections",
                "import operator",
                "import re",
                "import string",
                "import sys",
                "import unittest",
                "",
                "",
            ]
        )
        tail = ""
        if python_test_has_check_def(test_src) and not python_test_calls_check(test_src):
            tail = f"\n\ncheck({fn_name})\n"
        elif not python_test_has_check_def(test_src):
            test_calls = python_test_function_calls(test_src)
            if test_calls:
                tail = f"\n\n{test_calls}\n"
        return f"{py_imports}{full_code}\n\n{test_src}{tail}\n"

    if language == "c":
        headers = "\n".join(
            [
                "#include <assert.h>",
                "#include <math.h>",
                "#include <stdbool.h>",
                "#include <stdarg.h>",
                "#include <stdlib.h>",
                "#include <stdio.h>",
                "#include <string.h>",
                "",
            ]
        )
        full_code = collapse_duplicate_c_like_signature(full_code)
        full_code = prepare_c_like_source(full_code)
        test_src = prepare_c_like_source(test_src)
        return f"{headers}{full_code}\n\n{test_src}\n"

    if language == "cpp":
        # FIX2: 补全常用头文件，避免 std::string / std::vector 等找不到
        headers = "\n".join(
            [
                "#include <cassert>",
                "#include <cmath>",
                "#include <cstdio>",
                "#include <cstring>",
                "#include <string>",
                "#include <vector>",
                "#include <algorithm>",
                "#include <iostream>",
                "#include <sstream>",
                "#include <map>",
                "#include <set>",
                "#include <cstdarg>",
                "#include <bits/stdc++.h>",
                "using namespace std;",
                "",
            ]
        )
        full_code = collapse_duplicate_c_like_signature(full_code)
        full_code = prepare_c_like_source(full_code)
        test_src = prepare_c_like_source(test_src)
        return f"{headers}{full_code}\n\n{test_src}\n"

    if language == "java":
        # test_src 来自 mceval 数据集，末尾自带 class 的闭合 }，
        # 所以直接用 test_src 替换掉 wrap 末尾的 }，避免多一个 }
        imports = "\n".join([
            "import java.util.*;",
            "import java.util.Arrays;",
            "import java.lang.reflect.*;",
            "import java.math.*;",
            "import java.io.*;",
            "import java.text.*;",
            "",
        ])
        # 去掉 test_src 末尾多余的 }（数据集自带的 class 闭合括号）
        full_code = prepare_c_like_source(full_code)
        test_src = prepare_c_like_source(test_src)
        test_src_stripped = test_src.rstrip()
        if test_src_stripped.endswith("}"):
            # 最后一个 } 是 class 结尾，去掉它，由我们的 wrap 统一加
            last_brace = test_src_stripped.rfind("}")
            test_src_inner = test_src_stripped[:last_brace].rstrip()
        else:
            test_src_inner = test_src_stripped
        class_name = mceval_java_class_name(test_src)
        wrapped = "\n".join([
            imports,
            f"public class {class_name} {{",
            textwrap.indent(mceval_java_helper_fields(full_code), "    "),
            __import__("textwrap").indent(full_code, "    "),
            "",
            test_src_inner,
            "}",
        ])
        return wrapped
    if language == "csharp":
        test_src = prepare_mceval_csharp_test(test_src)
        full_code = prepare_c_like_source(full_code)
        test_src = prepare_c_like_source(test_src)
        test_src_stripped = test_src.rstrip()
        if test_src_stripped.endswith("}"):
            last_brace = test_src_stripped.rfind("}")
            test_src_inner = test_src_stripped[:last_brace].rstrip()
        else:
            test_src_inner = test_src_stripped
        wrapped = "\n".join(
            [
                "using System;",
                "using System.Collections.Generic;",
                "using System.Linq;",
                "using System.Text;",
                "using System.Text.RegularExpressions;",
                "using System.Numerics;",
                "",
                "class Problem",
                "{",
                textwrap.indent(full_code, "    "),
                "",
                textwrap.indent(test_src_inner, "    "),
                "}",
            ]
        )
        return wrapped

    if language == "go":
        wrapped = "\n".join(
            [
                "package main",
                "",
                "import (",
                '    "fmt"',
                '    "math"',
                '    "sort"',
                '    "strconv"',
                '    "strings"',
                '    "unicode"',
                ")",
                "",
                "// suppress unused import",
                "var _ = fmt.Sprintf",
                "var _ = math.Abs",
                "var _ = sort.Ints",
                "var _ = strconv.Itoa",
                "var _ = strings.Builder{}",
                "var _ = unicode.IsLetter",
                "",
                full_code,
                "",
            ]
        )
        return wrapped

    return f"{full_code}\n\n{test_src}\n"


def prepare_go_mceval_test(test_src: str) -> str:
    test_src = test_src.replace("assert := assert.New(t)", "assert := newAssert(t)")
    assert_helper = r'''
type localAssert struct {
    t *testing.T
}

func newAssert(t *testing.T) *localAssert {
    return &localAssert{t: t}
}

type assertPackage struct{}

var assert assertPackage

func (assertPackage) New(t *testing.T) *localAssert {
    return newAssert(t)
}

func (assertPackage) Equal(t *testing.T, expected interface{}, actual interface{}, msgAndArgs ...interface{}) {
    newAssert(t).Equal(expected, actual, msgAndArgs...)
}

func (assertPackage) InDelta(t *testing.T, expected interface{}, actual interface{}, delta float64, msgAndArgs ...interface{}) {
    newAssert(t).InDelta(expected, actual, delta, msgAndArgs...)
}

func (assertPackage) True(t *testing.T, value bool, msgAndArgs ...interface{}) {
    newAssert(t).True(value, msgAndArgs...)
}

func (assertPackage) False(t *testing.T, value bool, msgAndArgs ...interface{}) {
    newAssert(t).False(value, msgAndArgs...)
}

func (a *localAssert) Equal(expected interface{}, actual interface{}, msgAndArgs ...interface{}) {
    if !reflect.DeepEqual(expected, actual) {
        a.t.Fatalf("not equal: expected=%#v actual=%#v", expected, actual)
    }
}

func (a *localAssert) InDelta(expected interface{}, actual interface{}, delta float64, msgAndArgs ...interface{}) {
    exp, okExp := toFloat64(expected)
    act, okAct := toFloat64(actual)
    if !okExp || !okAct {
        a.t.Fatalf("InDelta expects numeric values: expected=%#v actual=%#v", expected, actual)
    }
    if math.Abs(exp-act) > delta {
        a.t.Fatalf("not in delta: expected=%#v actual=%#v delta=%#v", expected, actual, delta)
    }
}

func (a *localAssert) True(value bool, msgAndArgs ...interface{}) {
    if !value {
        a.t.Fatalf("expected true")
    }
}

func (a *localAssert) False(value bool, msgAndArgs ...interface{}) {
    if value {
        a.t.Fatalf("expected false")
    }
}

func toFloat64(value interface{}) (float64, bool) {
    switch v := value.(type) {
    case float64:
        return v, true
    case float32:
        return float64(v), true
    case int:
        return float64(v), true
    case int32:
        return float64(v), true
    case int64:
        return float64(v), true
    default:
        return 0, false
    }
}
'''
    return "\n".join(
        [
            "package main",
            "",
            "import (",
            '    "math"',
            '    "reflect"',
            '    "testing"',
            ")",
            "",
            assert_helper.strip(),
            "",
            test_src,
            "",
        ]
    )


def normalize_go_mceval_source(full_code: str) -> str:
    return re.sub(
        r"(?m)^(func\s+[A-Za-z_][A-Za-z0-9_]*\s*\([^)]*\)\s*(?:\([^)]*\)|[A-Za-z_][A-Za-z0-9_.*\[\]]*)?)\s*\n(?!\s*\{)",
        r"\1 {\n",
        full_code,
    )


def judge_mceval(language: str, full_code: str, row: dict[str, Any], timeout_sec: int) -> tuple[bool | None, str, str]:
    test_src = str(row.get("judge_payload", {}).get("test", ""))
    if not test_src.strip():
        return None, "unsupported", "missing mceval test"

    available, reason = toolchain_available(language)
    if not available:
        return None, "unsupported", reason

    # 从 fim_prompt prefix 里提取函数名（比从模型输出里提取更可靠）
    fim_prefix = decode_escaped_for_prompt(str(row.get("fim_prompt", ""))).split("<|fim_suffix|>", 1)[0].replace("<|fim_prefix|>", "", 1)
    fn_name_hint = (
        str(row.get("metadata", {}).get("entry_point", "")).strip()
        or extract_python_function_name(fim_prefix)
        or ""
    )
    if language == "go":
        full_code = normalize_go_mceval_source(full_code)
    program = wrap_mceval_code(language, full_code, test_src, fn_name_hint=fn_name_hint)

    with tempfile.TemporaryDirectory() as td:
        work = Path(td)

        if language == "python":
            py_path = work / "main.py"
            py_path.write_text(program, encoding="utf-8")
            rc, stdout, stderr, timed_out = run_cmd(["python", str(py_path)], timeout_sec=timeout_sec)
            if timed_out:
                return False, "timeout", stderr[:300]
            if rc == 0:
                return True, "ok", ""
            return False, "runtime_error", (stderr or stdout)[:300]

        if language == "c":
            c_path = work / "main.c"
            bin_path = work / "a.out"
            c_path.write_text(program, encoding="utf-8")
            rc, stdout, stderr, timed_out = run_cmd(
                ["gcc", str(c_path), "-lm", "-O2", "-std=c11", "-o", str(bin_path)], timeout_sec=timeout_sec
            )
            if timed_out:
                return False, "compile_timeout", stderr[:300]
            if rc != 0:
                return False, "compile_error", (stderr or stdout)[:300]
            rc, stdout, stderr, timed_out = run_cmd([str(bin_path)], timeout_sec=timeout_sec)
            if timed_out:
                return False, "timeout", stderr[:300]
            if rc == 0:
                return True, "ok", ""
            return False, "runtime_error", (stderr or stdout)[:300]

        if language == "cpp":
            cpp_path = work / "main.cpp"
            bin_path = work / "a.out"
            cpp_path.write_text(program, encoding="utf-8")
            rc, stdout, stderr, timed_out = run_cmd(
                ["g++", str(cpp_path), "-O2", "-std=c++17", "-o", str(bin_path)], timeout_sec=timeout_sec
            )
            if timed_out:
                return False, "compile_timeout", stderr[:300]
            if rc != 0:
                return False, "compile_error", (stderr or stdout)[:300]
            rc, stdout, stderr, timed_out = run_cmd([str(bin_path)], timeout_sec=timeout_sec)
            if timed_out:
                return False, "timeout", stderr[:300]
            if rc == 0:
                return True, "ok", ""
            return False, "runtime_error", (stderr or stdout)[:300]

        if language == "java":
            java_class = mceval_java_class_name(test_src)
            java_path = work / f"{java_class}.java"
            java_path.write_text(program, encoding="utf-8")
            rc, stdout, stderr, timed_out = run_cmd(["javac", str(java_path)], timeout_sec=timeout_sec)
            if timed_out:
                return False, "compile_timeout", stderr[:300]
            if rc != 0:
                return False, "compile_error", (stderr or stdout)[:300]
            rc, stdout, stderr, timed_out = run_cmd(["java", "-ea", "-cp", str(work), java_class], timeout_sec=timeout_sec)
            if timed_out:
                return False, "timeout", stderr[:300]
            if rc == 0:
                return True, "ok", ""
            return False, "runtime_error", (stderr or stdout)[:300]

        if language == "csharp":
            cs_path = work / "Problem.cs"
            exe_path = work / "Problem.exe"
            cs_path.write_text(program, encoding="utf-8")
            csharp_compiler = shutil.which("csc") or shutil.which("mcs")
            assert csharp_compiler is not None
            if Path(csharp_compiler).name == "csc":
                compile_cmd = [
                    csharp_compiler,
                    "-nologo",
                    "-langversion:latest",
                    "-r:System.Numerics.dll",
                    f"-out:{exe_path}",
                    str(cs_path),
                ]
            else:
                compile_cmd = [csharp_compiler, str(cs_path), f"/out:{exe_path}"]
            rc, stdout, stderr, timed_out = run_cmd(compile_cmd, timeout_sec=timeout_sec)
            if timed_out:
                return False, "compile_timeout", stderr[:300]
            if rc != 0:
                return False, "compile_error", (stderr or stdout)[:300]
            rc, stdout, stderr, timed_out = run_cmd(["mono", str(exe_path)], timeout_sec=timeout_sec)
            if timed_out:
                return False, "timeout", stderr[:300]
            if rc != 0:
                return False, "runtime_error", (stderr or stdout)[:300]
            lines = [ln.strip().lower() for ln in normalize_text(stdout).split("\n") if ln.strip()]
            if lines and all(ln == "true" for ln in lines):
                return True, "ok", ""
            if lines and any(ln == "false" for ln in lines):
                return False, "assert_failed", stdout[:300]
            return True, "ok", ""

        if language == "go":
            go_path = work / "main.go"
            test_path = work / "main_test.go"
            go_path.write_text(program, encoding="utf-8")
            test_path.write_text(prepare_go_mceval_test(test_src), encoding="utf-8")
            rc, stdout, stderr, timed_out = run_cmd(
                ["go", "test", str(go_path), str(test_path)],
                cwd=str(work),
                timeout_sec=timeout_sec,
            )
            if timed_out:
                return False, "timeout", (stderr or stdout)[:300]
            if rc != 0:
                if "syntax error" in stderr or "undefined" in stderr or "cannot" in stderr:
                    return False, "compile_error", stderr[:300]
                return False, "runtime_error", (stderr or stdout)[:300]
            return True, "ok", ""

    return None, "unsupported", f"unsupported mceval language: {language}"


def extract_java_public_class_name(source: str) -> str | None:
    m = re.search(r"\bpublic\s+(?:final\s+|abstract\s+)?class\s+([A-Za-z_][A-Za-z0-9_]*)\b", source)
    return m.group(1) if m else None


def extract_java_main_class_name(source: str) -> str | None:
    class_matches = list(re.finditer(r"\b(?:public\s+)?(?:final\s+|abstract\s+)?class\s+([A-Za-z_][A-Za-z0-9_]*)\b", source))
    if not class_matches:
        return None
    main_pos = source.find("public static void main")
    if main_pos < 0:
        main_pos = source.find("static public void main")
    if main_pos >= 0:
        before_main = [m for m in class_matches if m.start() < main_pos]
        if before_main:
            return before_main[-1].group(1)
    return class_matches[0].group(1)


def prepare_safim_java_source(full_code: str) -> tuple[str, str, str]:
    code = prepare_c_like_source(full_code)
    public_class = extract_java_public_class_name(code)
    main_class = public_class or extract_java_main_class_name(code) or "Main"
    if public_class:
        file_class = public_class
    else:
        file_class = main_class
        public_pattern = rf"\bpublic\s+class\s+{re.escape(main_class)}\b"
        if re.search(public_pattern, code):
            code = re.sub(public_pattern, f"public class {main_class}", code, count=1)
        else:
            code = re.sub(
                rf"\bclass\s+{re.escape(main_class)}\b",
                f"public class {main_class}",
                code,
                count=1,
            )
    return code, file_class, main_class


SAFIM_INPUT_FILES = ("input.txt", "in.txt")
SAFIM_OUTPUT_FILES = ("output.txt", "out.txt")


def prepare_safim_case_files(work: Path, case_input: str) -> None:
    for name in SAFIM_OUTPUT_FILES:
        path = work / name
        if path.exists():
            path.unlink()
    for name in SAFIM_INPUT_FILES:
        (work / name).write_text(case_input, encoding="utf-8")


def collect_safim_output(work: Path, stdout: str) -> str:
    if normalize_text(stdout):
        return stdout
    for name in SAFIM_OUTPUT_FILES:
        path = work / name
        if path.exists():
            return path.read_text(encoding="utf-8", errors="replace")
    return stdout


def judge_safim(language: str, full_code: str, row: dict[str, Any], timeout_sec: int) -> tuple[bool | None, str, str]:
    available, reason = toolchain_available(language)
    if not available:
        return None, "unsupported", reason

    unit_tests_raw = row.get("judge_payload", {}).get("unit_tests", "")
    if not unit_tests_raw:
        return None, "unsupported", "missing safim unit_tests"

    try:
        test_cases = json.loads(unit_tests_raw)
    except json.JSONDecodeError as exc:
        return None, "unsupported", f"invalid unit_tests json: {exc}"

    if not isinstance(test_cases, list) or len(test_cases) == 0:
        return None, "unsupported", "empty safim unit_tests"

    with tempfile.TemporaryDirectory() as td:
        work = Path(td)

        if language == "python":
            src_path = work / "main.py"
            src_path.write_text(escape_raw_newlines_in_c_like_literals(full_code), encoding="utf-8")
            run_cmd_base = ["python", str(src_path)]
            compile_ok = True
        elif language == "cpp":
            src_path = work / "main.cpp"
            bin_path = work / "a.out"
            src_path.write_text(prepare_c_like_source(full_code), encoding="utf-8")
            rc, stdout, stderr, timed_out = run_cmd(
                ["g++", "-include", "bits/stdc++.h", str(src_path), "-O2", "-std=c++17", "-o", str(bin_path)],
                timeout_sec=timeout_sec,
            )
            if timed_out:
                return False, "compile_timeout", stderr[:300]
            if rc != 0:
                return False, "compile_error", (stderr or stdout)[:300]
            run_cmd_base = [str(bin_path)]
            compile_ok = True
        elif language == "java":
            java_code, file_class, main_class = prepare_safim_java_source(full_code)
            src_path = work / f"{file_class}.java"
            src_path.write_text(java_code, encoding="utf-8")
            rc, stdout, stderr, timed_out = run_cmd(["javac", str(src_path)], timeout_sec=timeout_sec)
            if timed_out:
                return False, "compile_timeout", stderr[:300]
            if rc != 0:
                return False, "compile_error", (stderr or stdout)[:300]
            run_cmd_base = ["java", "-cp", str(work), main_class]
            compile_ok = True
        elif language == "csharp":
            src_path = work / "Program.cs"
            exe_path = work / "Problem.exe"
            src_path.write_text(prepare_c_like_source(full_code), encoding="utf-8")
            csharp_compiler = shutil.which("csc") or shutil.which("mcs")
            assert csharp_compiler is not None
            if Path(csharp_compiler).name == "csc":
                compile_cmd = [
                    csharp_compiler,
                    "-nologo",
                    "-langversion:latest",
                    "-r:System.Numerics.dll",
                    f"-out:{exe_path}",
                    str(src_path),
                ]
            else:
                compile_cmd = [csharp_compiler, str(src_path), f"/out:{exe_path}"]
            rc, stdout, stderr, timed_out = run_cmd(compile_cmd, timeout_sec=timeout_sec)
            if timed_out:
                return False, "compile_timeout", stderr[:300]
            if rc != 0:
                return False, "compile_error", (stderr or stdout)[:300]
            run_cmd_base = ["mono", str(exe_path)]
            compile_ok = True
        else:
            compile_ok = False
            run_cmd_base = []

        if not compile_ok:
            return None, "unsupported", f"unsupported safim language: {language}"

        for idx, case in enumerate(test_cases):
            case_input = str(case.get("input", ""))
            expected_list = case.get("output", [])
            expected = str(expected_list[0] if expected_list else "")
            prepare_safim_case_files(work, case_input)

            rc, stdout, stderr, timed_out = run_cmd(
                run_cmd_base,
                cwd=str(work),
                timeout_sec=timeout_sec,
                stdin_text=case_input,
            )
            if timed_out:
                return False, "timeout", f"case={idx}; {stderr[:200]}"
            if rc != 0:
                return False, "runtime_error", f"case={idx}; {(stderr or stdout)[:200]}"

            stdout = collect_safim_output(work, stdout)
            got_norm = normalize_text(stdout)
            exp_norm = normalize_text(expected)
            if not outputs_match(stdout, expected):
                return False, "wrong_answer", f"case={idx}; got={got_norm[:120]!r}; expected={exp_norm[:120]!r}"

    return True, "ok", ""


def judge_candidate(
    row: dict[str, Any],
    predicted_completion: str,
    timeout_sec: int,
) -> tuple[bool | None, str, str]:
    try:
        return _judge_candidate_impl(row, predicted_completion, timeout_sec)
    except Exception as exc:
        return False, "judge_error", f"{type(exc).__name__}: {str(exc)[:240]}"


def _judge_candidate_impl(
    row: dict[str, Any],
    predicted_completion: str,
    timeout_sec: int,
) -> tuple[bool | None, str, str]:
    source_dataset = str(row.get("source_dataset", "unknown")).lower()
    language = str(row.get("language", "unknown")).lower()

    if language not in SUPPORTED_LANGUAGES:
        return None, "unsupported", f"unsupported language: {language}"

    # FIX Bug2: fim_prompt 里的 prefix/suffix 也需要 unescape
    fim_prompt_raw = str(row.get("fim_prompt", ""))
    fim_prompt = decode_escaped_for_prompt(fim_prompt_raw)
    prefix, suffix = parse_fim_prompt(fim_prompt)
    full_code = f"{prefix}{predicted_completion}{suffix}"

    if source_dataset == "humaneval":
        if language != "python":
            return None, "unsupported", "humaneval currently expected python"
        return judge_humaneval_python(full_code, row, timeout_sec)

    if source_dataset == "mceval":
        return judge_mceval(language, full_code, row, timeout_sec)

    if source_dataset == "safim":
        return judge_safim(language, full_code, row, timeout_sec)

    return None, "unsupported", f"unsupported dataset: {source_dataset}"


