import inspect
def _find_the_caller_in_the_stack(name:str=None, type:type = None):
        caller_frame = inspect.currentframe().f_back
        while caller_frame:
            caller_locals = caller_frame.f_locals
            caller_instance = caller_locals.get("self", None)

            if name==caller_instance.__class__.__name__ or type==caller_instance.__class__:
                return caller_instance
            caller_frame = caller_frame.f_back


def _is_primitive_type(val):
    return val.__class__.__module__=="builtins" and not type(val) is type