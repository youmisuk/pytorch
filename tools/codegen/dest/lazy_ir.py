from typing import List, Optional, Union
import itertools
from typing_extensions import Literal
from dataclasses import dataclass
import textwrap

from tools.codegen.context import method_with_native_function, native_function_manager
from tools.codegen.utils import Target, mapMaybe
from tools.codegen.model import (BaseType, OptionalType, DispatchKey, NativeFunction,
                                 NativeFunctionsGroup, SchemaKind,
                                 TensorOptionsArguments,
                                 DeviceCheckType, Argument, assert_never,
                                 is_cuda_dispatch_key, BackendIndex,
                                 gets_generated_out_inplace_wrapper)
from tools.codegen.api.types import (BaseTy, BaseCppType, BaseCType, OptionalCType,
                                     Binding, ConstRefCType,
                                     CppSignature, CppSignatureGroup,
                                     Expr, MutRefCType, kernel_signature,
                                     NativeSignature, tensorT, NamedCType,
                                     DispatcherSignature)
import tools.codegen.api.meta as meta
import tools.codegen.api.cpp as cpp
import tools.codegen.api.structured as structured
from tools.codegen.api.translate import translate
from tools.codegen.selective_build.selector import SelectiveBuilder

# Generates {backend}_lazy_ir.h and .cpp
#


@dataclass(frozen=True)
class LazyIR:
    backend_index: BackendIndex

    target: Union[
        Literal[Target.DEFINITION],
        Literal[Target.DECLARATION],
    ]

    # TODO(whc) probably use selector instead of building a separate index for ops to codegen
    # Selector object to determine which operators to generate
    # registration code for.
    selector: SelectiveBuilder

    # The namespace that the kernels are written in. This is just `at::native` for in-tree kernels.
    cpp_namespace: str

    @method_with_native_function
    def __call__(self, f: Union[NativeFunctionsGroup, NativeFunction]) -> List[str]:
        if isinstance(f, NativeFunctionsGroup):
            func = f.functional.func
        else:
            func = f.func
        # print(func.name)
        if func.name in self.backend_index.index:
            return self.gen(f)
        else:
            return []

    def gen(self, f: Union[NativeFunctionsGroup, NativeFunction]) -> List[str]:
        # for now, we just want one IR class decl and soon after also the method defs
        # and we use the functional version not out/inplace.
        func = f.functional.func if isinstance(f, NativeFunctionsGroup) else f.func
        class_name = str(func.name).lower().capitalize()

        """IR Value inputs are IR values that stand in for Tensors in the function schema.

           Lets assume for now that we just want to pass through optional Tensors as
           optional Values and let the lowerings handle the rest.
        """
        valueT = BaseCppType('ir', 'Value')

        def value_inputs():
            """This just filters the function args to find the Tensors (incl optionals),
            and makes them into Value types
            """
            ret = []
            for t in func.arguments.post_self_positional:
                if t.is_write:
                    assert False, "Do we expect mutable tensors in lazytensor bindings?"
                else:
                    if isinstance(t.type, BaseType) and t.type.name == BaseTy.Tensor:
                        ret.append(NamedCType(t.name, ConstRefCType(BaseCType(valueT))))
                    elif isinstance(t.type, OptionalType) and str(t.type.elem) == 'Tensor':
                        ret.append(NamedCType(t.name, ConstRefCType(OptionalCType(BaseCType(valueT)))))
                    else:
                        # assert False, f"didn't get to lists or others yet, {t}"
                        pass
            return ret

        return [f"""\
class {class_name} : public LazyNodeBase {{
public:
{class_name}({", ".join([f"{i.cpp_type()} {i.name}" for i in value_inputs()])})
    : LazyNodeBase(ir::OpKind(at::aten::{func.name.name})),
            {{{", ".join([f"{i.name}" for i in value_inputs()])}}},
            /*num_outputs=*/{len(func.returns)},
            lazy_tensors::util::MHash(/*TODO*/),
    dtype_(dtype)
{{}}

std::string ToString() const override;

NodePtr Clone(OpList operands) const override;

private:
std::vector<c10::Optional<ir::Value>> inputs;

}}
""", ]
