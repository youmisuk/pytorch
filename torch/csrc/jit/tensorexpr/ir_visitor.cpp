#include <torch/csrc/jit/tensorexpr/ir_visitor.h>

#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/reduction.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>

#include <c10/util/irange.h>

namespace torch {
namespace jit {
namespace tensorexpr {

template <
    typename Op,
    typename std::enable_if<std::is_same<
        decltype(detail::bin_op_deducer(std::declval<Op>())),
        void>::value>::type* = nullptr>
static void visit_binary_op(NodePtr<Op> v, IRVisitor* visitor) {
  v->lhs()->accept(visitor);
  v->rhs()->accept(visitor);
}

void IRVisitor::visit(AddPtr v) {
  visit_binary_op(v, this);
}

void IRVisitor::visit(SubPtr v) {
  visit_binary_op(v, this);
}

void IRVisitor::visit(MulPtr v) {
  visit_binary_op(v, this);
}

void IRVisitor::visit(DivPtr v) {
  visit_binary_op(v, this);
}

void IRVisitor::visit(ModPtr v) {
  visit_binary_op(v, this);
}

void IRVisitor::visit(MaxPtr v) {
  visit_binary_op(v, this);
}

void IRVisitor::visit(MinPtr v) {
  visit_binary_op(v, this);
}

void IRVisitor::visit(AndPtr v) {
  visit_binary_op(v, this);
}

void IRVisitor::visit(OrPtr v) {
  visit_binary_op(v, this);
}

void IRVisitor::visit(XorPtr v) {
  visit_binary_op(v, this);
}

void IRVisitor::visit(LshiftPtr v) {
  visit_binary_op(v, this);
}

void IRVisitor::visit(RshiftPtr v) {
  visit_binary_op(v, this);
}

void IRVisitor::visit(CompareSelectPtr v) {
  v->lhs()->accept(this);
  v->rhs()->accept(this);
  v->ret_val1()->accept(this);
  v->ret_val2()->accept(this);
}

// NOLINTNEXTLINE
#define IMM_VISIT(Type, Name) \
  void IRVisitor::visit(Name##ImmPtr v) {}
AT_FORALL_SCALAR_TYPES_AND2(Bool, Half, IMM_VISIT);
#undef IMM_VISIT

void IRVisitor::visit(CastPtr v) {
  v->src_value()->accept(this);
}
void IRVisitor::visit(BitCastPtr v) {
  v->src_value()->accept(this);
}
void IRVisitor::visit(VarPtr v) {}

void IRVisitor::visit(RampPtr v) {
  v->base()->accept(this);
  v->stride()->accept(this);
}

void IRVisitor::visit(LoadPtr v) {
  v->buf()->accept(this);
  for (ExprPtr ind : v->indices()) {
    ind->accept(this);
  }
}

void IRVisitor::visit(BufPtr v) {
  v->base_handle()->accept(this);
}

void IRVisitor::visit(StorePtr v) {
  v->buf()->accept(this);
  for (ExprPtr ind : v->indices()) {
    ind->accept(this);
  }
  v->value()->accept(this);
}

void IRVisitor::visit(AtomicAddPtr v) {
  v->buf()->accept(this);
  for (ExprPtr ind : v->indices()) {
    ind->accept(this);
  }
  v->value()->accept(this);
}

void IRVisitor::visit(SyncThreadsPtr v) {}

void IRVisitor::visit(ExternalCallPtr v) {
  v->buf()->accept(this);
  for (BufPtr buf_arg : v->buf_args()) {
    buf_arg->accept(this);
  }
  for (ExprPtr arg : v->args()) {
    arg->accept(this);
  }
}

void IRVisitor::visit(BlockPtr v) {
  for (StmtPtr s : *v) {
    s->accept(this);
  }
}

void IRVisitor::visit(ForPtr v) {
  v->var()->accept(this);
  v->start()->accept(this);
  v->stop()->accept(this);
  if (v->body()) {
    v->body()->accept(this);
  }
}

void IRVisitor::visit(BroadcastPtr v) {
  v->value()->accept(this);
}

void IRVisitor::visit(IfThenElsePtr v) {
  v->condition()->accept(this);
  v->true_value()->accept(this);
  v->false_value()->accept(this);
}

void IRVisitor::visit(IntrinsicsPtr v) {
  for (const auto i : c10::irange(v->nparams())) {
    v->param(i)->accept(this);
  }
}

void IRVisitor::visit(AllocatePtr v) {
  v->buffer_var()->accept(this);
  std::vector<ExprPtr> dims = v->dims();
  for (ExprPtr dim : dims) {
    dim->accept(this);
  }
}

void IRVisitor::visit(FreePtr v) {
  v->buffer_var()->accept(this);
}

void IRVisitor::visit(LetPtr v) {
  v->var()->accept(this);
  v->value()->accept(this);
}

void IRVisitor::visit(CondPtr v) {
  ExprPtr condition = v->condition();
  StmtPtr true_stmt = v->true_stmt();
  StmtPtr false_stmt = v->false_stmt();
  condition->accept(this);
  if (true_stmt) {
    true_stmt->accept(this);
  }
  if (false_stmt) {
    false_stmt->accept(this);
  }
}

void IRVisitor::visit(TermPtr v) {
  v->scalar()->accept(this);
  for (auto t : v->variables()) {
    t->accept(this);
  }
}

void IRVisitor::visit(PolynomialPtr v) {
  v->scalar()->accept(this);
  for (auto t : v->variables()) {
    t->accept(this);
  }
}

void IRVisitor::visit(RoundOffPtr v) {
  v->lhs()->accept(this);
  v->rhs()->accept(this);
}

void IRVisitor::visit(MaxTermPtr v) {
  if (v->scalar()) {
    v->scalar()->accept(this);
  }
  for (auto t : v->variables()) {
    t->accept(this);
  }
}

void IRVisitor::visit(MinTermPtr v) {
  if (v->scalar()) {
    v->scalar()->accept(this);
  }
  for (auto t : v->variables()) {
    t->accept(this);
  }
}

void IRVisitor::visit(ReduceOpPtr v) {
  v->body()->accept(this);

  for (auto r : v->reduce_args()) {
    r->accept(this);
  }
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch
