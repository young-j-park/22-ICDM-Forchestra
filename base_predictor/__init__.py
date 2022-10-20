
from args import BaseModelArguments
from base_predictor.base import BasePredictors
from base_predictor.lstm import LSTMForecasting


def build_base_model(args: BaseModelArguments) -> BasePredictors:
    if args.model_name == 'lstm':
        return LSTMForecasting(**vars(args))
    elif args.model_name == 'bilstm':
        return LSTMForecasting(bidirectional=True, **vars(args))
    else:
        raise ValueError
