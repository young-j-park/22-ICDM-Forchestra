
from args import ModelArguments
from model.base import BaseModel
from model.linear import LinearMetaLearner
from model.static import StaticMetaLearner


def build_model(args: ModelArguments) -> BaseModel:
    if args.meta_model_name == 'linear':
        return LinearMetaLearner(**vars(args))
    elif args.meta_model_name == 'static':
        return StaticMetaLearner(**vars(args))
    else:
        raise ValueError
