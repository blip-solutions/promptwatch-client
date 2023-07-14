import datetime
from decimal import Decimal
import inspect
import types
def find_the_caller_in_the_stack(name:str=None, type:type = None):
        caller_frame = inspect.currentframe().f_back
        while caller_frame:
            caller_locals = caller_frame.f_locals
            caller_instance = caller_locals.get("self", None)

            if (name and name==caller_instance.__class__.__name__) or (type and isinstance(caller_instance, type)):
                return caller_instance
            caller_frame = caller_frame.f_back


def is_primitive_type(type_val):
    return type_val in [int, float, str, bool, Decimal, type(None)]


def copy_dict_serializable_values(dict_value):
    res={}
    if isinstance(dict_value, dict):
        for key, value in dict_value.items():
            if  is_primitive_type(type(value)):
                res[key] = value
            elif isinstance(value, dict):
                res[key] = copy_dict_serializable_values(value)
            elif isinstance(value, list):
                res[key] = copy_list_serializable_values(value)
            elif isinstance(value, datetime.datetime):
                res[key] = value.isoformat()
        return res
    else:
        raise ValueError(f"Expected dict. Got: {dict_value}")

def copy_list_serializable_values(list_value):
    if isinstance(list_value, list):
        value=[]
        for i, item in enumerate(list_value):
            if is_primitive_type(item):
                value[i] = item
            elif isinstance(item, list):
                value[i] = copy_list_serializable_values(item)
            elif isinstance(item, dict):
                value[i] = copy_dict_serializable_values(item)
            elif isinstance(value, datetime.datetime):
                value[i] = value.isoformat()
        return value
    else:
        raise ValueError(f"Expected list. Got: {value}")





def wrap_a_method(object, function_name, decorator):
      # need to go around pydantic restrictions on __setattr__  by using __dict__ directly
      original_func = getattr(object.__class__, function_name)
      object.__dict__[function_name] = types.MethodType(decorator(original_func),object)
      

class classproperty:
    def __init__(self, fget):
        self.fget = classmethod(fget)

    def __get__(self, instance, owner):
        return self.fget.__get__(None, owner)()
