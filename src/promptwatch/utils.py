import inspect
import types
def find_the_caller_in_the_stack(name:str=None, type:type = None):
        caller_frame = inspect.currentframe().f_back
        while caller_frame:
            caller_locals = caller_frame.f_locals
            caller_instance = caller_locals.get("self", None)

            if name==caller_instance.__class__.__name__ or isinstance(caller_instance, type):
                return caller_instance
            caller_frame = caller_frame.f_back


def is_primitive_type(val):
    return val.__class__.__module__=="builtins" and not type(val) is type and not type(val) is dict


def wrap_a_method(object, function_name, decorator):
      # need to go around pydantic restrictions on __setattr__  by using __dict__ directly
      original_func = getattr(object.__class__, function_name)
      object.__dict__[function_name] = types.MethodType(decorator(original_func),object)
      

class classproperty:
    def __init__(self, fget):
        self.fget = classmethod(fget)

    def __get__(self, instance, owner):
        return self.fget.__get__(None, owner)()
