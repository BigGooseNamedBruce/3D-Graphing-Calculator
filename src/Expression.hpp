#pragma once

// =============================================================================
// A small expression compiler for surface equations: height = f(x, y, t).
//
// Parsing is recursive descent, but it emits a flat postfix instruction list
// rather than an AST. The evaluator runs ~5x per vertex (the mesh generator
// takes central differences for normals), so at 160x160 that is ~128k
// evaluations per snapshot — a pointer-chasing tree of virtual eval() calls is
// measurably worse than a contiguous vector walked with a small stack.
//
// Provides:
//   Expression   the compiled program; eval(x, y, t) runs it on a fixed stack
//   Parser       Parser::parse(source) -> ParseResult
//   ParseResult  ok flag, the Expression, and on failure a message plus the
//                byte offset of the offending token
//
// Grammar, one method per level:
//   expr  := term (('+' | '-') term)*
//   term  := unary (('*' | '/') unary | implicit-multiply)*
//   unary := ('-' | '+') unary | power        looser than '^', so -x^2 = -(x^2)
//   power := atom ('^' unary)?                right-associative, so 2^3^2 = 512
//   atom  := number | 'x' | 'y' | 't' | pi | tau | e | call | '(' expr ')'
//
// Implicit multiplication means "2x", "3(x+1)" and "x y" parse the way they do
// on a calculator rather than erroring.
//
// Parsing never throws. Failures come back inside ParseResult so the caller can
// keep rendering the previous surface and show the message in the UI, which is
// what makes live editing tolerable — a half-typed expression must not blank
// the view.
//
// Stack depth is computed during the parse and checked against
// Expression::kStackDepth, so eval() cannot overflow its fixed-size stack for
// any program this file produced.
// =============================================================================

#include <cctype>
#include <cmath>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace expr {

enum class Op : uint8_t {
    Const, VarX, VarY, VarT,
    Add, Sub, Mul, Div, Pow, Neg,
    Fn1, Fn2,
};

enum class F1 : uint8_t {
    Sin, Cos, Tan, Asin, Acos, Atan, Sinh, Cosh, Tanh,
    Exp, Log, Log2, Log10, Sqrt, Abs, Floor, Ceil, Round, Sign,
};

enum class F2 : uint8_t { Atan2, Min, Max, Pow, Hypot };

struct Instr {
    Op op;
    uint8_t fn = 0; // F1 or F2 ordinal, depending on op
    float value = 0.0f; // Const only
};

class Expression {
public:
    // Empty until a successful parse; evaluates to 0.
    float eval(float x, float y, float t) const {
        float stack[kStackDepth];
        int sp = 0;

        for (const Instr& in : code_) {
            switch (in.op) {
                case Op::Const: stack[sp++] = in.value; break;
                case Op::VarX: stack[sp++] = x; break;
                case Op::VarY: stack[sp++] = y; break;
                case Op::VarT: stack[sp++] = t; break;

                case Op::Neg: stack[sp - 1] = -stack[sp - 1]; break;

                case Op::Add: --sp; stack[sp - 1] += stack[sp]; break;
                case Op::Sub: --sp; stack[sp - 1] -= stack[sp]; break;
                case Op::Mul: --sp; stack[sp - 1] *= stack[sp]; break;
                case Op::Div: --sp; stack[sp - 1] /= stack[sp]; break;
                case Op::Pow: --sp; stack[sp - 1] = std::pow(stack[sp - 1], stack[sp]); break;

                case Op::Fn1: stack[sp - 1] = apply1(static_cast<F1>(in.fn), stack[sp - 1]); break;
                case Op::Fn2: --sp; stack[sp - 1] = apply2(static_cast<F2>(in.fn),
                                                             stack[sp - 1], stack[sp]); break;
            }
        }
        return sp > 0 ? stack[sp - 1] : 0.0f;
    }

    bool empty() const { return code_.empty(); }

    // Whether the program reads t, i.e. whether the surface animates. A surface
    // that does not cannot change while the camera holds still, which is what
    // lets the renderer skip regenerating it.
    bool usesTime() const {
        for (const Instr& in : code_) {
            if (in.op == Op::VarT) {
                return true;
            }
        }
        return false;
    }

    // Worst-case stack use, computed at parse time. Guarantees eval() cannot
    // overflow its fixed-size stack for any expression this class produced.
    static constexpr int kStackDepth = 64;

private:
    friend class Parser;
    std::vector<Instr> code_;

    static float apply1(F1 f, float v) {
        switch (f) {
            case F1::Sin: return std::sin(v);
            case F1::Cos: return std::cos(v);
            case F1::Tan: return std::tan(v);
            case F1::Asin: return std::asin(v);
            case F1::Acos: return std::acos(v);
            case F1::Atan: return std::atan(v);
            case F1::Sinh: return std::sinh(v);
            case F1::Cosh: return std::cosh(v);
            case F1::Tanh: return std::tanh(v);
            case F1::Exp: return std::exp(v);
            case F1::Log: return std::log(v);
            case F1::Log2: return std::log2(v);
            case F1::Log10: return std::log10(v);
            case F1::Sqrt: return std::sqrt(v);
            case F1::Abs: return std::fabs(v);
            case F1::Floor: return std::floor(v);
            case F1::Ceil: return std::ceil(v);
            case F1::Round: return std::round(v);
            case F1::Sign: return v > 0.0f ? 1.0f : (v < 0.0f ? -1.0f : 0.0f);
        }
        return 0.0f;
    }

    static float apply2(F2 f, float a, float b) {
        switch (f) {
            case F2::Atan2: return std::atan2(a, b);
            case F2::Min: return std::fmin(a, b);
            case F2::Max: return std::fmax(a, b);
            case F2::Pow: return std::pow(a, b);
            case F2::Hypot: return std::hypot(a, b);
        }
        return 0.0f;
    }
};

struct ParseResult {
    bool ok = false;
    Expression expression;
    std::string error; // human-readable, empty when ok
    size_t position = 0; // byte offset of the offending token
};

class Parser {
public:
    static ParseResult parse(const std::string& src) {
        Parser p(src);
        ParseResult r;

        p.skipSpace();
        if (p.atEnd()) {
            r.error = "empty expression";
            return r;
        }

        p.parseExpr();
        if (!p.error_.empty()) {
            r.error = p.error_;
            r.position = p.errorPos_;
            return r;
        }

        p.skipSpace();
        if (!p.atEnd()) {
            r.error = std::string("unexpected '") + p.src_[p.pos_] + "'";
            r.position = p.pos_;
            return r;
        }
        if (p.maxDepth_ >= Expression::kStackDepth) {
            r.error = "expression too deeply nested";
            return r;
        }

        r.ok = true;
        r.expression.code_ = std::move(p.code_);
        return r;
    }

private:
    explicit Parser(const std::string& s) : src_(s) {}

    const std::string& src_;
    size_t pos_ = 0;
    std::vector<Instr> code_;
    std::string error_;
    size_t errorPos_ = 0;
    int depth_ = 0; // current stack depth while emitting
    int maxDepth_ = 0;

    bool atEnd() const { return pos_ >= src_.size(); }
    char peek() const { return atEnd() ? '\0' : src_[pos_]; }

    void skipSpace() { while (!atEnd() && std::isspace(static_cast<unsigned char>(src_[pos_]))) ++pos_; }

    void fail(const std::string& msg, size_t at) {
        if (error_.empty()) {
            error_ = msg;
            errorPos_ = at;
        }
    }

    void push(int n) { depth_ += n; if (depth_ > maxDepth_) maxDepth_ = depth_; }

    void emit(Op op, uint8_t fn = 0, float v = 0.0f) {
        code_.push_back(Instr{op, fn, v});
        switch (op) {
            case Op::Const: case Op::VarX: case Op::VarY: case Op::VarT: push(1); break;
            case Op::Neg: case Op::Fn1: break; // in-place
            default: push(-1); break; // binary ops consume one
        }
    }

    // expr := term (('+' | '-') term)*
    void parseExpr() {
        parseTerm();
        for (;;) {
            skipSpace();
            const char c = peek();
            if (c != '+' && c != '-') {
                return;
            }
            ++pos_;
            parseTerm();
            if (!error_.empty()) {
                return;
            }
            emit(c == '+' ? Op::Add : Op::Sub);
        }
    }

    // term := unary (('*' | '/') unary | implicit-multiply)*
    void parseTerm() {
        parseUnary();
        for (;;) {
            skipSpace();
            if (!error_.empty()) {
                return;
            }
            const char c = peek();
            if (c == '*' || c == '/') {
                ++pos_;
                parseUnary();
                if (!error_.empty()) {
                    return;
                }
                emit(c == '*' ? Op::Mul : Op::Div);
            } else if (startsAtom(c)) {
                // Implicit multiplication, so "2x", "3(x+1)" and "x y" work the
                // way they do in a calculator rather than being parse errors.
                parseUnary();
                if (!error_.empty()) {
                    return;
                }
                emit(Op::Mul);
            } else {
                return;
            }
        }
    }

    static bool startsAtom(char c) {
        return std::isdigit(static_cast<unsigned char>(c)) ||
               std::isalpha(static_cast<unsigned char>(c)) || c == '_' || c == '(' || c == '.';
    }

    // unary := ('-' | '+') unary | power     -- looser than '^', so -x^2 == -(x^2)
    void parseUnary() {
        skipSpace();
        const char c = peek();
        if (c == '-') {
            ++pos_;
            parseUnary();
            if (error_.empty()) {
                emit(Op::Neg);
            }
            return;
        }
        if (c == '+') {
            ++pos_;
            parseUnary();
            return;
        }
        parsePower();
    }

    // power := atom ('^' unary)?            -- right-associative
    void parsePower() {
        parseAtom();
        if (!error_.empty()) {
            return;
        }
        skipSpace();
        if (peek() == '^') {
            ++pos_;
            parseUnary();
            if (!error_.empty()) {
                return;
            }
            emit(Op::Pow);
        }
    }

    void parseAtom() {
        skipSpace();
        if (atEnd()) {
            fail("unexpected end of expression", pos_);
            return;
        }

        const char c = peek();

        if (c == '(') {
            ++pos_;
            parseExpr();
            if (!error_.empty()) {
                return;
            }
            skipSpace();
            if (peek() != ')') {
                fail("expected ')'", pos_);
                return;
            }
            ++pos_;
            return;
        }

        if (std::isdigit(static_cast<unsigned char>(c)) || c == '.') {
            const size_t start = pos_;
            while (!atEnd() && (std::isdigit(static_cast<unsigned char>(src_[pos_])) || src_[pos_] == '.')) {
                ++pos_;
            }
            // Accept an exponent only when it is well-formed, so "2e" stays
            // "2 * e" (Euler's number) rather than becoming a malformed literal.
            if (!atEnd() && (src_[pos_] == 'e' || src_[pos_] == 'E')) {
                const size_t save = pos_;
                size_t look = pos_ + 1;
                if (look < src_.size() && (src_[look] == '+' || src_[look] == '-')) {
                    ++look;
                }
                if (look < src_.size() && std::isdigit(static_cast<unsigned char>(src_[look]))) {
                    pos_ = look;
                    while (!atEnd() && std::isdigit(static_cast<unsigned char>(src_[pos_]))) {
                        ++pos_;
                    }
                } else {
                    pos_ = save;
                }
            }
            emit(Op::Const, 0, std::stof(src_.substr(start, pos_ - start)));
            return;
        }

        if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
            const size_t start = pos_;
            while (!atEnd() && (std::isalnum(static_cast<unsigned char>(src_[pos_])) || src_[pos_] == '_')) {
                ++pos_;
            }
            std::string name = src_.substr(start, pos_ - start);
            for (char& ch : name) {
                ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
            }

            skipSpace();
            const bool call = (peek() == '(');
            if (call) {
                return parseCall(name, start);
            }

            if (name == "x") {
                emit(Op::VarX);
                return;
            }
            if (name == "y") {
                emit(Op::VarY);
                return;
            }
            if (name == "t") {
                emit(Op::VarT);
                return;
            }
            if (name == "pi") {
                emit(Op::Const, 0, 3.14159265358979f);
                return;
            }
            if (name == "tau") {
                emit(Op::Const, 0, 6.28318530717959f);
                return;
            }
            if (name == "e") {
                emit(Op::Const, 0, 2.71828182845905f);
                return;
            }

            fail("unknown name '" + name + "'", start);
            return;
        }

        fail(std::string("unexpected '") + c + "'", pos_);
    }

    void parseCall(const std::string& name, size_t namePos) {
        ++pos_; // consume '('
        parseExpr();
        if (!error_.empty()) {
            return;
        }

        bool twoArg = false;
        skipSpace();
        if (peek() == ',') {
            ++pos_;
            parseExpr();
            if (!error_.empty()) {
                return;
            }
            twoArg = true;
            skipSpace();
        }
        if (peek() != ')') {
            fail("expected ')'", pos_);
            return;
        }
        ++pos_;

        uint8_t id;
        if (twoArg) {
            if (!lookupF2(name, id)) {
                fail("'" + name + "' does not take 2 arguments", namePos);
                return;
            }
            emit(Op::Fn2, id);
        } else {
            if (!lookupF1(name, id)) {
                fail("unknown function '" + name + "'", namePos);
                return;
            }
            emit(Op::Fn1, id);
        }
    }

    static bool lookupF1(const std::string& n, uint8_t& out) {
        static const std::pair<const char*, F1> table[] = {
            {"sin", F1::Sin}, {"cos", F1::Cos}, {"tan", F1::Tan},
            {"asin", F1::Asin}, {"acos", F1::Acos}, {"atan", F1::Atan},
            {"sinh", F1::Sinh}, {"cosh", F1::Cosh}, {"tanh", F1::Tanh},
            {"exp", F1::Exp}, {"ln", F1::Log}, {"log", F1::Log10},
            {"log2", F1::Log2}, {"log10", F1::Log10},
            {"sqrt", F1::Sqrt}, {"abs", F1::Abs}, {"floor", F1::Floor},
            {"ceil", F1::Ceil}, {"round", F1::Round}, {"sign", F1::Sign},
        };
        for (const auto& e : table) {
            if (n == e.first) {
                out = static_cast<uint8_t>(e.second);
                return true;
            }
        }
        return false;
    }

    static bool lookupF2(const std::string& n, uint8_t& out) {
        static const std::pair<const char*, F2> table[] = {
            {"atan2", F2::Atan2}, {"min", F2::Min}, {"max", F2::Max},
            {"pow", F2::Pow}, {"hypot", F2::Hypot},
        };
        for (const auto& e : table) {
            if (n == e.first) {
                out = static_cast<uint8_t>(e.second);
                return true;
            }
        }
        return false;
    }
};

} // namespace expr
