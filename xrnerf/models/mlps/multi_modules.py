import math

import kilonerf_cuda
import torch
import torch.nn.functional as F
from torch import nn


# Only this function had to be changed to account for multi networks (weight tensors have additionally a network dimension)
def _calculate_fan_in_and_fan_out(tensor):
    fan_in = tensor.size(-1)
    fan_out = tensor.size(-2)
    return fan_in, fan_out


def _calculate_correct_fan(tensor, mode):
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out']
    if mode not in valid_modes:
        raise ValueError('Mode {} not supported, please use one of {}'.format(
            mode, valid_modes))

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == 'fan_in' else fan_out


def calculate_gain(nonlinearity, param=None):
    linear_fns = [
        'linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d',
        'conv_transpose2d', 'conv_transpose3d'
    ]
    if nonlinearity in linear_fns or nonlinearity == 'sigmoid':
        return 1
    elif nonlinearity == 'tanh':
        return 5.0 / 3
    elif nonlinearity == 'relu':
        return math.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        if param is None:
            negative_slope = 0.01
        elif not isinstance(param, bool) and isinstance(
                param, int) or isinstance(param, float):
            # True/False are instances of int, hence check above
            negative_slope = param
        else:
            raise ValueError(
                'negative_slope {} not a valid number'.format(param))
        return math.sqrt(2.0 / (1 + negative_slope**2))
    else:
        raise ValueError('Unsupported nonlinearity {}'.format(nonlinearity))


def kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(
        3.0) * std  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)


def kaiming_normal_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    with torch.no_grad():
        return tensor.normal_(0, std)


def xavier_uniform_(tensor, gain=1.):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    a = math.sqrt(
        3.0) * std  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        return tensor.uniform_(-a, a)


def xavier_normal_(tensor, gain=1.):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    with torch.no_grad():
        return tensor.normal_(0., std)


class Sine(nn.Module):
    def __init__(self, w0=1.):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


# For hard parameter sharing
class SharedLinear(nn.Module):
    __constants__ = ['in_features', 'out_features']

    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 nonlinearity='leaky_relu',
                 weight_initialization_method='kaiming_uniform',
                 bias_initialization_method='standard'):
        super(SharedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.nonlinearity = nonlinearity
        self.weight_initialization_method = weight_initialization_method
        self.bias_initialization_method = bias_initialization_method
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.weight_initialization_method == 'kaiming_uniform':
            nn.init.kaiming_uniform_(self.weight,
                                     a=math.sqrt(5),
                                     nonlinearity=self.nonlinearity)
        elif self.weight_initialization_method == 'kaiming_normal':
            nn.init.kaiming_normal_(self.weight,
                                    a=math.sqrt(5),
                                    nonlinearity=self.nonlinearity)
        elif self.weight_initialization_method == 'xavier_uniform':
            nn.init.xavier_uniform_(self.weight,
                                    gain=nn.init.calculate_gain(
                                        self.nonlinearity))
        elif self.weight_initialization_method == 'xavier_normal':
            nn.init.xavier_normal_(self.weight,
                                   gain=nn.init.calculate_gain(
                                       self.nonlinearity))
        if self.bias is not None:
            if self.bias_initialization_method == 'standard':
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)
            elif self.bias_initialization_method == 'zeros':
                nn.init.zeros_(self.bias)

    # batch_size_per_network is a dummy argument
    def forward(self, input, batch_size_per_network=None):
        has_network_dim = len(list(input.size())) == 3
        if has_network_dim:  # ignore network dimension
            num_networks = input.size(0)
            input = input.view(-1, self.in_features)
        out = F.linear(input, self.weight, self.bias)
        if has_network_dim:
            out = out.view(num_networks, -1, self.out_features)
        return out


def naive_multimatmul(biases, input_vectors, weights, out_features,
                      in_features, batch_size_per_network):
    num_points = len(input_vectors)
    num_networks = len(biases)
    result_naive = torch.empty(num_points,
                               out_features,
                               device=torch.device('cuda'))
    start_index = 0
    for network_index in range(num_networks):
        end_index = start_index + batch_size_per_network[network_index].item()
        #torch.matmul(input_vectors[start_index:end_index], weights[network_index], out=result_naive[start_index:end_index])
        torch.addmm(biases[network_index],
                    input_vectors[start_index:end_index],
                    weights[network_index],
                    out=result_naive[start_index:end_index])
        start_index = end_index
    return result_naive


def naive_multimatmul_differentiable(biases, input_vectors, weights,
                                     out_features, in_features,
                                     batch_size_per_network):
    num_points = len(input_vectors)
    num_networks = len(biases)
    result_naive = torch.empty(num_points,
                               out_features,
                               device=torch.device('cuda'))
    start_index = 0
    for network_index in range(num_networks):
        end_index = start_index + batch_size_per_network[network_index].item()
        temp_res = torch.addmm(biases[network_index],
                               input_vectors[start_index:end_index],
                               weights[network_index])
        result_naive[start_index:end_index] = temp_res
        start_index = end_index
    return result_naive


class AddMultiMatMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, biases, input_vectors, weights, out_features, in_features,
                batch_size_per_network, group_limits, aux_index,
                aux_index_backward):
        ctx.save_for_backward(biases, input_vectors, weights,
                              batch_size_per_network)
        ctx.out_features = out_features
        ctx.in_features = in_features
        ctx.group_limits = group_limits
        ctx.aux_index = aux_index
        ctx.aux_index_backward = aux_index_backward
        return kilonerf_cuda.multimatmul_magma_grouped_static(
            biases, input_vectors, weights, out_features, in_features,
            batch_size_per_network, 4, 1024, group_limits, aux_index)

    @staticmethod
    def backward(ctx, grad_output):
        biases, input_vectors, weights, batch_size_per_network = ctx.saved_tensors

        grad_output = grad_output.contiguous()

        grad_biases = None
        grad_input_vectors = None
        grad_weights = None

        grad_biases = kilonerf_cuda.multi_row_sum_reduction(
            grad_output, batch_size_per_network)

        grad_input_vectors = kilonerf_cuda.multimatmul_magma_grouped_static_without_bias_transposed_weights(
            biases, grad_output, weights, ctx.in_features, ctx.out_features,
            batch_size_per_network, 4, 1024, ctx.group_limits,
            ctx.aux_index_backward)

        grad_weights = kilonerf_cuda.multimatmul_A_transposed(
            input_vectors, grad_output, batch_size_per_network)

        return grad_biases, grad_input_vectors, grad_weights, None, None, None, None, None, None


class MultiNetworkLinear(nn.Module):
    rng_state = None

    def __init__(self,
                 num_networks,
                 in_features,
                 out_features,
                 nonlinearity='leaky_relu',
                 bias=True,
                 implementation='bmm',
                 nonlinearity_params=None,
                 use_same_initialization_for_all_networks=False,
                 network_rng_seed=None,
                 weight_initialization_method='kaiming_uniform',
                 bias_initialization_method='standard'):

        super(MultiNetworkLinear, self).__init__()
        self.num_networks = num_networks
        self.in_features = in_features
        self.out_features = out_features
        self.implementation = implementation
        self.use_same_initialization_for_all_networks = use_same_initialization_for_all_networks
        self.network_rng_seed = network_rng_seed
        # weight is created in reset_parameters()
        if self.implementation.startswith('multimatmul'):
            self.group_limits = [2048, 1024]  # tunable
            self.aux_index = kilonerf_cuda.init_multimatmul_magma_grouped(
                self.num_networks, self.out_features, self.in_features,
                self.group_limits)
            if self.implementation == 'multimatmul_differentiable':
                # out_features and in_features are interchanged
                self.aux_index_backward = kilonerf_cuda.init_multimatmul_magma_grouped(
                    self.num_networks, self.in_features, self.out_features,
                    self.group_limits)
        self.nonlinearity = nonlinearity
        self.nonlinearity_params = nonlinearity_params
        self.weight_initialization_method = weight_initialization_method
        self.bias_initialization_method = bias_initialization_method
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_networks, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        self.weight = nn.Parameter(
            torch.Tensor(self.num_networks, self.out_features,
                         self.in_features))

        # Use a separate RNG seed for network initialization to be able to keep
        # other random aspects (i.e. batch sampling) fixed, while varying network initialization
        if self.network_rng_seed is not None:
            previous_rng_state = torch.random.get_rng_state()
            if MultiNetworkLinear.rng_state is None:
                torch.random.manual_seed(self.network_rng_seed)
            else:
                torch.random.set_rng_state(MultiNetworkLinear.rng_state)

        if self.nonlinearity != 'sine':
            if self.weight_initialization_method == 'kaiming_uniform':
                kaiming_uniform_(self.weight,
                                 a=math.sqrt(5),
                                 nonlinearity=self.nonlinearity)
            elif self.weight_initialization_method == 'kaiming_normal':
                kaiming_normal_(self.weight,
                                a=math.sqrt(5),
                                nonlinearity=self.nonlinearity)
            elif self.weight_initialization_method == 'xavier_uniform':
                xavier_uniform_(self.weight,
                                gain=calculate_gain(self.nonlinearity))
            elif self.weight_initialization_method == 'xavier_normal':
                xavier_normal_(self.weight,
                               gain=calculate_gain(self.nonlinearity))
            if self.bias is not None:
                if self.bias_initialization_method == 'standard':
                    fan_in, _ = _calculate_fan_in_and_fan_out(self.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(self.bias, -bound, bound)
                elif self.bias_initialization_method == 'zeros':
                    nn.init.zeros_(self.bias)
        else:  # For SIREN
            c, w0, is_first = self.nonlinearity_params[
                'c'], self.nonlinearity_params['w0'], self.nonlinearity_params[
                    'is_first']
            w_std = (1 / self.in_features) if is_first else (
                math.sqrt(c / self.in_features) / w0)
            nn.init.uniform_(self.weight, -w_std, w_std)
            if self.bias is not None:
                nn.init.uniform_(self.bias, -w_std, w_std)

        if self.network_rng_seed is not None:
            MultiNetworkLinear.rng_state = torch.random.get_rng_state()
            torch.random.set_rng_state(previous_rng_state)

        if self.use_same_initialization_for_all_networks:
            with torch.no_grad():
                self.weight[1:] = self.weight[0]
                self.bias[1:] = self.bias[0]

        if 'multimatmul' in self.implementation:
            self.weight.data = self.weight.data.view(
                self.num_networks, self.in_features,
                self.out_features).contiguous()

    def forward(self, x, batch_size_per_network=None, bias=None, weight=None):
        # For testing purposes override weight and bias
        if bias is not None:
            self.bias = bias
        if weight is not None:
            self.weight = weight
        if self.implementation == 'multimatmul':
            # x = num_points x in_features
            return kilonerf_cuda.multimatmul_magma_grouped_static(
                self.bias, x.contiguous(), self.weight, self.out_features,
                self.in_features, batch_size_per_network, 4, 1024,
                self.group_limits, self.aux_index)
        elif self.implementation == 'multimatmul_differentiable':
            return AddMultiMatMul.apply(self.bias, x.contiguous(), self.weight,
                                        self.out_features, self.in_features,
                                        batch_size_per_network,
                                        self.group_limits, self.aux_index,
                                        self.aux_index_backward)
        elif self.implementation == 'naive_multimatmul_differentiable':
            return naive_multimatmul_differentiable(self.bias, x, self.weight,
                                                    self.out_features,
                                                    self.in_features,
                                                    batch_size_per_network)
        else:
            # x = num_networks x batch_size x in_features
            batch_size = x.size(1)
            if self.num_networks > 1:
                if self.implementation == 'bmm':
                    weight_transposed = self.weight.permute(
                        0, 2, 1)  # num_networks x in_features x out_features

                    # num_networks x batch_size x in_features @ num_networks x in_features x out_features = num_networks x batch_size x out_features
                    product = torch.bmm(x, weight_transposed)
                    bias_view = self.bias.unsqueeze(1)
                elif self.implementation == 'matmul':
                    input_view = x.unsqueeze(
                        3)  # num_networks x batch_size x in_features x 1
                    weight_view = self.weight.unsqueeze(
                        1)  # num_networks x 1 x out_features x in_features
                    product = torch.matmul(weight_view, input_view).squeeze(
                        3)  # num_networks x batch_size x out_features
                    bias_view = self.bias.unsqueeze(
                        1)  # num_networks x 1 x out_features
                result = product + bias_view  # (num_networks * batch_size) x out_features
            else:
                input_view = x.squeeze(0)
                weight_view = self.weight.squeeze(0)
                bias_view = self.bias.squeeze(0)
                result = F.linear(input_view, weight_view, bias_view)
            return result.view(self.num_networks, batch_size,
                               self.out_features)


def extract_linears(network):
    linears, shared_linears = [], []
    for module in network.modules():
        if isinstance(module, MultiNetworkLinear):
            linears.append(module)
        if isinstance(module, SharedLinear):
            shared_linears.append(module)
    return linears, shared_linears


class MultiNetwork(nn.Module):
    def __init__(self,
                 num_networks,
                 num_position_channels,
                 num_direction_channels,
                 num_output_channels,
                 hidden_layer_size,
                 num_hidden_layers,
                 refeed_position_index=None,
                 late_feed_direction=False,
                 direction_layer_size=None,
                 nonlinearity='relu',
                 nonlinearity_initalization='pass_leaky_relu',
                 use_single_net=False,
                 linear_implementation='bmm',
                 use_same_initialization_for_all_networks=False,
                 network_rng_seed=None,
                 weight_initialization_method='kaiming_uniform',
                 bias_initialization_method='standard',
                 alpha_rgb_initalization='updated_yenchenlin',
                 use_hard_parameter_sharing_for_color=False,
                 view_dependent_dropout_probability=-1,
                 use_view_independent_color=False):
        super(MultiNetwork, self).__init__()

        self.num_networks = num_networks
        self.num_position_channels = num_position_channels
        self.num_direction_channels = num_direction_channels
        self.num_output_channels = num_output_channels
        self.hidden_layer_size = hidden_layer_size
        self.num_hidden_layers = num_hidden_layers
        self.refeed_position_index = refeed_position_index
        self.late_feed_direction = late_feed_direction
        self.direction_layer_size = direction_layer_size
        self.nonlinearity = nonlinearity
        self.nonlinearity_initalization = nonlinearity_initalization  # 'pass_leaky_relu', 'pass_actual_nonlinearity'
        self.use_single_net = use_single_net
        self.linear_implementation = linear_implementation
        self.use_same_initialization_for_all_networks = use_same_initialization_for_all_networks
        self.network_rng_seed = network_rng_seed
        self.weight_initialization_method = weight_initialization_method
        self.bias_initialization_method = bias_initialization_method
        self.alpha_rgb_initalization = alpha_rgb_initalization  # 'updated_yenchenlin', 'pass_actual_nonlinearity'
        self.use_hard_parameter_sharing_for_color = use_hard_parameter_sharing_for_color
        self.view_dependent_dropout_probability = view_dependent_dropout_probability
        self.use_view_independent_color = use_view_independent_color

        nonlinearity_params = {}
        if nonlinearity == 'sigmoid':
            self.activation = nn.Sigmoid()
        if nonlinearity == 'tanh':
            self.activation = nn.Tanh()
        if nonlinearity == 'relu':
            self.activation = nn.ReLU()
        if nonlinearity == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        if nonlinearity == 'sine':
            nonlinearity_params = {'w0': 30., 'c': 6., 'is_first': True}
            self.activation = Sine(nonlinearity_params['w0'])

        # TODO: weight_initialization_method and bias_initialization_method are been ignored
        def linear_layer(in_features,
                         out_features,
                         actual_nonlinearity,
                         use_hard_parameter_sharing=False):
            if self.nonlinearity_initalization == 'pass_actual_nonlinearity':  # proper way of doing things
                passed_nonlinearity = actual_nonlinearity
            elif self.nonlinearity_initalization == 'pass_leaky_relu':  # to reproduce the old behaviour (doesn't make a lot of sense though)
                passed_nonlinearity = 'leaky_relu'
            if not use_hard_parameter_sharing:
                return MultiNetworkLinear(
                    self.num_networks,
                    in_features,
                    out_features,
                    nonlinearity=passed_nonlinearity,
                    nonlinearity_params=nonlinearity_params,
                    implementation=linear_implementation,
                    use_same_initialization_for_all_networks=
                    use_same_initialization_for_all_networks,
                    network_rng_seed=network_rng_seed)
            else:
                print('Using hard parameter sharing')
                return SharedLinear(in_features,
                                    out_features,
                                    bias=True,
                                    nonlinearity=passed_nonlinearity)

        if self.late_feed_direction:
            self.pts_linears = [
                linear_layer(self.num_position_channels,
                             self.hidden_layer_size, self.nonlinearity)
            ]
            nonlinearity_params = nonlinearity_params.copy().update(
                {'is_first': False})
            for i in range(self.num_hidden_layers - 1):
                if i == self.refeed_position_index:
                    new_layer = linear_layer(
                        self.hidden_layer_size + self.num_position_channels,
                        self.hidden_layer_size, self.nonlinearity)
                else:
                    new_layer = linear_layer(self.hidden_layer_size,
                                             self.hidden_layer_size,
                                             self.nonlinearity)
                self.pts_linears.append(new_layer)
            self.pts_linears = nn.ModuleList(self.pts_linears)
            self.direction_layer = linear_layer(
                self.num_direction_channels + self.hidden_layer_size,
                self.direction_layer_size, self.nonlinearity,
                self.use_hard_parameter_sharing_for_color)

            if self.use_view_independent_color:
                feature_output_size = self.hidden_layer_size + 4  # + RGBA
            else:
                feature_output_size = self.hidden_layer_size
            self.feature_linear = linear_layer(self.hidden_layer_size,
                                               feature_output_size, 'linear')
            # In the updated yenchenlin implementation which follows now closely the original tensorflow implementation
            # 'linear' is passed to these two layers, but it also makes sense to pass the actual nonlinearites here
            if not self.use_view_independent_color:
                self.alpha_linear = linear_layer(
                    self.hidden_layer_size, 1,
                    'linear' if self.alpha_rgb_initalization
                    == 'updated_yenchenlin' else 'relu')
            self.rgb_linear = linear_layer(
                self.direction_layer_size, 3,
                'linear' if self.alpha_rgb_initalization
                == 'updated_yenchenlin' else 'sigmoid',
                self.use_hard_parameter_sharing_for_color)

            self.view_dependent_parameters = list(
                self.direction_layer.parameters()
            ) + list(
                self.rgb_linear.parameters()
            )  # needed for L2 regularization only on the view-dependent part of the network

            if self.view_dependent_dropout_probability > 0:
                self.dropout_after_feature = nn.Dropout(
                    self.view_dependent_dropout_probability)
                self.dropout_after_direction_layer = nn.Dropout(
                    self.view_dependent_dropout_probability)

        else:
            layers = [
                linear_layer(
                    self.num_position_channels + self.num_direction_channels,
                    self.hidden_layer_size), self.activation
            ]
            nonlinearity_params = nonlinearity_params.copy().update(
                {'is_first': False})
            for _ in range(
                    self.num_hidden_layers
            ):  # TODO: should be also self.num_hidden_layers - 1
                layers += [
                    linear_layer(self.hidden_layer_size,
                                 self.hidden_layer_size), self.activation
                ]
            layers += [
                linear_layer(self.hidden_layer_size, self.num_output_channels)
            ]
            self.layers = nn.Sequential(*layers)

    # needed for fused kernel
    def serialize_params(self):
        # fused kernel expects IxO matrix instead of OxI matrix
        def process_weight(w):
            return w.reshape(self.num_networks, -1)

        self.serialized_params = []
        for l in self.pts_linears:
            self.serialized_params += [l.bias, process_weight(l.weight)]

        self.serialized_params.append(
            torch.cat([self.alpha_linear.bias, self.feature_linear.bias],
                      dim=1))
        self.serialized_params.append(
            process_weight(
                torch.cat(
                    [self.alpha_linear.weight, self.feature_linear.weight],
                    dim=2)))
        for l in [self.direction_layer, self.rgb_linear]:
            self.serialized_params += [l.bias, process_weight(l.weight)]
        self.serialized_params = torch.cat(self.serialized_params,
                                           dim=1).contiguous()

    # random_directions will be used for regularizing the view-independent color
    def forward(self, x, batch_size_per_network=None, random_directions=None):
        if self.late_feed_direction:
            if isinstance(x, list):
                positions, directions = x
                # frees memory of inputs
                x[0] = None
                x[1] = None
            else:
                positions, directions = torch.split(
                    x,
                    [self.num_position_channels, self.num_direction_channels],
                    dim=-1)
            h = positions
            for i, l in enumerate(self.pts_linears):
                h = self.pts_linears[i](h, batch_size_per_network)
                h = self.activation(h)
                if i == self.refeed_position_index:
                    h = torch.cat([positions, h], -1)
            del positions
            if not self.use_view_independent_color:
                alpha = self.alpha_linear(h, batch_size_per_network)
            feature = self.feature_linear(
                h, batch_size_per_network
            )  # TODO: investigate why they don't use an activation function on top of feature layer!
            if self.view_dependent_dropout_probability > 0:
                feature = self.dropout_after_feature(feature)
            if self.use_view_independent_color:
                rgb_view_independent, alpha, feature = torch.split(
                    feature, [3, 1, self.hidden_layer_size], dim=-1)
            del h

            # Regularizing the view-independent color to be the mean of view-dependent colors sampled at some random directions
            if random_directions is not None:
                assert self.use_view_independent_color == True, 'this regularization only makes sense if we output a view-independent color'
                num_random_directions = random_directions.size(0)
                batch_size = feature.size(0)
                feature_size = feature.size(1)
                feature = feature.repeat(1, num_random_directions + 1).view(
                    -1, feature_size)
                random_directions = random_directions.repeat(
                    batch_size, 1).view(batch_size, num_random_directions, -1)
                directions = torch.cat(
                    [directions.unsqueeze(1), random_directions],
                    dim=1).view(batch_size * (num_random_directions + 1), -1)
                batch_size_per_network = (num_random_directions +
                                          1) * batch_size_per_network

            # View-dependent part of the network:
            h = torch.cat([feature, directions], -1)
            del feature
            del directions
            h = self.direction_layer(h, batch_size_per_network)
            h = self.activation(h)
            if self.view_dependent_dropout_probability > 0:
                h = self.dropout_after_direction_layer(h)

            rgb = self.rgb_linear(h, batch_size_per_network)
            del h

            if self.use_view_independent_color:
                if random_directions is None:
                    rgb = rgb + rgb_view_independent
                else:
                    mean_rgb = rgb.view(batch_size, num_random_directions + 1,
                                        3)
                    mean_rgb = mean_rgb + rgb_view_independent.unsqueeze(1)
                    rgb = mean_rgb[:, 0]
                    mean_rgb = mean_rgb.mean(dim=1)
                    mean_regularization_term = torch.abs(
                        mean_rgb - rgb_view_independent).mean()
                    del mean_rgb
                del rgb_view_independent

            result = torch.cat([rgb, alpha], -1)

            if random_directions is not None:
                return result, mean_regularization_term
            else:
                return result
        else:
            return self.layers(x)

    def extract_single_network(self, network_index):
        single_network = MultiNetwork(
            1,
            self.num_position_channels,
            self.num_direction_channels,
            self.num_output_channels,
            self.hidden_layer_size,
            self.num_hidden_layers,
            self.refeed_position_index,
            self.late_feed_direction,
            self.direction_layer_size,
            self.nonlinearity,
            self.nonlinearity_initalization,
            self.use_single_net,
            use_hard_parameter_sharing_for_color=self.
            use_hard_parameter_sharing_for_color,
            view_dependent_dropout_probability=self.
            view_dependent_dropout_probability,
            use_view_independent_color=self.use_view_independent_color)

        multi_linears, multi_shared_linears = extract_linears(self)
        single_linears, single_shared_linears = extract_linears(single_network)
        with torch.no_grad():
            for single_linear, multi_linear in zip(single_linears,
                                                   multi_linears):
                single_linear.weight.data[0] = multi_linear.weight.data[
                    network_index]
                single_linear.bias.data[0] = multi_linear.bias.data[
                    network_index]

            for single_shared_linear, multi_shared_linear in zip(
                    single_shared_linears, multi_shared_linears):
                single_shared_linear.weight.data = multi_shared_linear.weight.data
                single_shared_linear.bias.data = multi_shared_linear.bias.data

        return single_network
