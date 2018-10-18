from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.base_layer import InputSpec
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.layers.recurrent import _standardize_args
from tensorflow.python.keras.layers.recurrent import RNN
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.utils import conv_utils

class STLSTM2D(RNN):
  """Base class for convolutional-recurrent layers.
  Arguments:
    cell: A RNN cell instance. A RNN cell is a class that has:
        - a `call(input_at_t, states_at_t)` method, returning
            `(output_at_t, states_at_t_plus_1)`. The call method of the
            cell can also take the optional argument `constants`, see
            section "Note on passing external constants" below.
        - a `state_size` attribute. This can be a single integer
            (single state) in which case it is
            the number of channels of the recurrent state
            (which should be the same as the number of channels of the cell
            output). This can also be a list/tuple of integers
            (one size per state). In this case, the first entry
            (`state_size[0]`) should be the same as
            the size of the cell output.
    return_sequences: Boolean. Whether to return the last output.
        in the output sequence, or the full sequence.
    return_state: Boolean. Whether to return the last state
        in addition to the output.
    go_backwards: Boolean (default False).
        If True, process the input sequence backwards and return the
        reversed sequence.
    stateful: Boolean (default False). If True, the last state
        for each sample at index i in a batch will be used as initial
        state for the sample of index i in the following batch.
    input_shape: Use this argument to specify the shape of the
        input when this layer is the first one in a model.
  Input shape:
    5D tensor with shape:
    `(samples, timesteps, channels, rows, cols)`
    if data_format='channels_first' or 5D tensor with shape:
    `(samples, timesteps, rows, cols, channels)`
    if data_format='channels_last'.
  Output shape:
    - if `return_state`: a list of tensors. The first tensor is
        the output. The remaining tensors are the last states,
        each 5D tensor with shape:
        `(samples, timesteps, filters, new_rows, new_cols)`
        if data_format='channels_first'
        or 5D tensor with shape:
        `(samples, timesteps, new_rows, new_cols, filters)`
        if data_format='channels_last'.
        `rows` and `cols` values might have changed due to padding.
    - if `return_sequences`: 5D tensor with shape:
        `(samples, timesteps, filters, new_rows, new_cols)`
        if data_format='channels_first'
        or 5D tensor with shape:
        `(samples, timesteps, new_rows, new_cols, filters)`
        if data_format='channels_last'.
    - else, 4D tensor with shape:
        `(samples, filters, new_rows, new_cols)`
        if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, new_rows, new_cols, filters)`
        if data_format='channels_last'.
  Masking:
    This layer supports masking for input data with a variable number
    of timesteps. To introduce masks to your data,
    use an Embedding layer with the `mask_zero` parameter
    set to `True`.
  Note on using statefulness in RNNs:
    You can set RNN layers to be 'stateful', which means that the states
    computed for the samples in one batch will be reused as initial states
    for the samples in the next batch. This assumes a one-to-one mapping
    between samples in different successive batches.
    To enable statefulness:
        - specify `stateful=True` in the layer constructor.
        - specify a fixed batch size for your model, by passing
             - if sequential model:
                `batch_input_shape=(...)` to the first layer in your model.
             - if functional model with 1 or more Input layers:
                `batch_shape=(...)` to all the first layers in your model.
                This is the expected shape of your inputs
                *including the batch size*.
                It should be a tuple of integers,
                e.g. `(32, 10, 100, 100, 32)`.
                Note that the number of rows and columns should be specified
                too.
        - specify `shuffle=False` when calling fit().
    To reset the states of your model, call `.reset_states()` on either
    a specific layer, or on your entire model.
  Note on specifying the initial state of RNNs:
    You can specify the initial state of RNN layers symbolically by
    calling them with the keyword argument `initial_state`. The value of
    `initial_state` should be a tensor or list of tensors representing
    the initial state of the RNN layer.
    You can specify the initial state of RNN layers numerically by
    calling `reset_states` with the keyword argument `states`. The value of
    `states` should be a numpy array or list of numpy arrays representing
    the initial state of the RNN layer.
  Note on passing external constants to RNNs:
    You can pass "external" constants to the cell using the `constants`
    keyword argument of `RNN.__call__` (as well as `RNN.call`) method. This
    requires that the `cell.call` method accepts the same keyword argument
    `constants`. Such constants can be used to condition the cell
    transformation on additional static inputs (not changing over time),
    a.k.a. an attention mechanism.
  """

  def __init__(self,
               cell,
               return_sequences=False,
               return_state=False,
               go_backwards=False,
               stateful=False,
               unroll=False,
               **kwargs):
    if unroll:
      raise TypeError('Unrolling isn\'t possible with '
                      'STLSTMs.')

    super(STLSTM2D, self).__init__(cell,
                                   return_sequences,
                                   return_state,
                                   go_backwards,
                                   stateful,
                                   unroll,
                                   **kwargs)
    self.input_spec = [InputSpec(ndim=5)]
    self.states = None
    self._num_constants = None
    self._stackedcells = False
    if isinstance(self.cell, StackedSTLSTMCells):
      self._stackedcells = True

  @tf_utils.shape_type_conversion
  def compute_output_shape(self, input_shape):
    single_shape = tuple(self.cell.output_size.as_list()[1:])
    output_shape = input_shape[:2] + single_shape # [batch, time, <single_shape>]

    if not self.return_sequences: # only the last output is returned
      output_shape = output_shape[:1] + output_shape[2:] # [batch, <single_shape>]

    if self.return_state: # [[outputshape], (batch, <single_shape>) for "m, hN, cN..., "h1", "c1"]
      output_shape = [output_shape]
      output_shape += [tuple([input_shape[0]] + single_shape)
                       for _ in range(1 + 2 * len(self.cell.cells))]
    return output_shape

  @tf_utils.shape_type_conversion
  def build(self, input_shape):
    # Note input_shape will be list of shapes of initial states and
    # constants if these are passed in __call__.
    if self._num_constants is not None:
      constants_shape = input_shape[-self._num_constants:]  # pylint: disable=E1130
    else:
      constants_shape = None

    if isinstance(input_shape, list):
      input_shape = input_shape[0]

    batch_size = input_shape[0] if self.stateful else None
    self.input_spec[0] = InputSpec(shape=(batch_size, None) + input_shape[2:5])

    # allow cell (if layer) to build before we set or validate state_spec
    if isinstance(self.cell, Layer):
      step_input_shape = (input_shape[0],) + input_shape[2:]
      if constants_shape is not None:
        self.cell.build([step_input_shape] + constants_shape)
      else:
        self.cell.build(step_input_shape)

    # set or validate state_spec
    if hasattr(self.cell.state_size, '__len__'):
      state_size = list(self.cell.state_size)
    else:
      state_size = [self.cell.state_size]

    if self.cell.data_format == 'channels_first':
      self.state_spec = [InputSpec(shape=(None, dim, None, None))
                         for dim in state_size]
    elif self.cell.data_format == 'channels_last':
      self.state_spec = [InputSpec(shape=(None, None, None, dim))
                           for dim in state_size]
    if self.stateful:
      self.reset_states()
    self.built = True

  def get_initial_state(self, inputs):
    initial_states = []
    first = True
    if self._stackedcells:
      for cell in self.cell.cells:
        shape = list(cell.kernel_shape)
        shape[-1] = cell.filters
        if first: # Make m, h, c states
          initial_state = K.zeros_like(inputs)
          initial_state = K.sum(initial_state, axis=1)
          initial_state = cell.input_conv(initial_state,
                                          K.zeros(tuple(shape)),
                                          padding=cell.padding)
          initial_states += [initial_state for _ in range(3)]
          first = False
        else: # if not first make h, c states
          initial_state = K.zeros_like(initial_state)
          initial_state = cell.input_conv(initial_state,
                                          K.zeros(tuple(shape)),
                                          padding=cell.padding)
          initial_states += [initial_state for _ in range(2)]
    
    else: # Single cell
      shape = list(self.cell.kernel_shape)
      shape[-1] = self.cell.filters
      initial_state = K.zeros_like(inputs)
      initial_state = self.cell.inputs_conv(initial_state,
                                            K.zeros(tuple(shape)),
                                            padding=self.cell.padding)
      initial_states += [initial_state for _ in range(3)]
    return initial_states

  def __call__(self, inputs, initial_state=None, constants=None, **kwargs):
    inputs, initial_state, constants = _standardize_args(
        inputs, initial_state, constants, self._num_constants)

    if initial_state is None and constants is None:
      return super(STLSTM2D, self).__call__(inputs, **kwargs)

    # If any of `initial_state` or `constants` are specified and are Keras
    # tensors, then add them to the inputs and temporarily modify the
    # input_spec to include them.

    additional_inputs = []
    additional_specs = []
    if initial_state is not None:
      kwargs['initial_state'] = initial_state
      additional_inputs += initial_state
      self.state_spec = []
      for state in initial_state:
        shape = K.int_shape(state)
        self.state_spec.append(InputSpec(shape=shape))

      additional_specs += self.state_spec
    if constants is not None:
      kwargs['constants'] = constants
      additional_inputs += constants
      self.constants_spec = [InputSpec(shape=K.int_shape(constant))
                             for constant in constants]
      self._num_constants = len(constants)
      additional_specs += self.constants_spec
    # at this point additional_inputs cannot be empty
    for tensor in additional_inputs:
      if K.is_keras_tensor(tensor) != K.is_keras_tensor(additional_inputs[0]):
        raise ValueError('The initial state or constants of an RNN'
                         ' layer cannot be specified with a mix of'
                         ' Keras tensors and non-Keras tensors')

    if K.is_keras_tensor(additional_inputs[0]):
      # Compute the full input spec, including state and constants
      full_input = [inputs] + additional_inputs
      full_input_spec = self.input_spec + additional_specs
      # Perform the call with temporarily replaced input_spec
      original_input_spec = self.input_spec
      self.input_spec = full_input_spec
      output = super(STLSTM2D, self).__call__(full_input, **kwargs)
      self.input_spec = original_input_spec
      return output
    else:
      return super(STLSTM2D, self).__call__(inputs, **kwargs)

  def call(self,
           inputs,
           mask=None,
           training=None,
           initial_state=None,
           constants=None):
    # note that the .build() method of subclasses MUST define
    # self.input_spec and self.state_spec with complete input shapes.
    if isinstance(inputs, list):
      inputs = inputs[0]
    if initial_state is not None:
      pass
    elif self.stateful:
      initial_state = self.states
    else:
      initial_state = self.get_initial_state(inputs)

    if isinstance(mask, list):
      mask = mask[0]
    if len(initial_state) != len(self.states):
      raise ValueError('Layer has ' + str(len(self.states)) +
                       ' states but was passed ' +
                       str(len(initial_state)) +
                       ' initial states.')
    timesteps = K.int_shape(inputs)[1]

    kwargs = {}
    if generic_utils.has_arg(self.cell.call, 'training'):
      kwargs['training'] = training

    if constants:
      if not generic_utils.has_arg(self.cell.call, 'constants'):
        raise ValueError('RNN cell does not support constants')

      def step(inputs, states):
        constants = states[-self._num_constants:]
        states = states[:-self._num_constants]
        return self.cell.call(inputs, states, constants=constants,
                              **kwargs)
    else:
      def step(inputs, states):
        return self.cell.call(inputs, states, **kwargs)

    last_output, outputs, states = K.rnn(step,
                                         inputs,
                                         initial_state,
                                         constants=constants,
                                         go_backwards=self.go_backwards,
                                         mask=mask,
                                         input_length=timesteps)
    if self.stateful:
      updates = []
      for i in range(len(states)):
        updates.append(K.update(self.states[i], states[i]))
      self.add_update(updates, inputs=True)

    if self.return_sequences:
      output = outputs
    else:
      output = last_output

    # Properly set learning phase
    if getattr(last_output, '_uses_learning_phase', False):
      output._uses_learning_phase = True

    if self.return_state:
      if not isinstance(states, (list, tuple)):
        states = [states]
      else:
        states = list(states)
      return [output] + states
    else:
      return output

  def reset_states(self, states=None):
    if not self.stateful:
      raise AttributeError('Layer must be stateful.')
    input_shape = self.input_spec[0].shape
    state_shape = self.compute_output_shape(input_shape)
    if self.return_state:
      state_shape = state_shape[0]
    if self.return_sequences:
      state_shape = state_shape[:1].concatenate(state_shape[2:])
    if None in state_shape:
      raise ValueError('If a RNN is stateful, it needs to know '
                       'its batch size. Specify the batch size '
                       'of your input tensors: \n'
                       '- If using a Sequential model, '
                       'specify the batch size by passing '
                       'a `batch_input_shape` '
                       'argument to your first layer.\n'
                       '- If using the functional API, specify '
                       'the time dimension by passing a '
                       '`batch_shape` argument to your Input layer.\n'
                       'The same thing goes for the number of rows and '
                       'columns.')

    # helper function
    def get_tuple_shape(nb_channels):
      result = list(state_shape)
      if self.cell.data_format == 'channels_first':
        result[1] = nb_channels
      elif self.cell.data_format == 'channels_last':
        result[3] = nb_channels
      else:
        raise KeyError
      return tuple(result)

    # initialize state if None
    if self.states[0] is None:
      if hasattr(self.cell.state_size, '__len__'):
        self.states = [K.zeros(get_tuple_shape(dim))
                       for dim in self.cell.state_size]
      else:
        self.states = [K.zeros(get_tuple_shape(self.cell.state_size))]
    elif states is None:
      if hasattr(self.cell.state_size, '__len__'):
        for state, dim in zip(self.states, self.cell.state_size):
          K.set_value(state, np.zeros(get_tuple_shape(dim)))
      else:
        K.set_value(self.states[0],
                    np.zeros(get_tuple_shape(self.cell.state_size)))
    else:
      if not isinstance(states, (list, tuple)):
        states = [states]
      if len(states) != len(self.states):
        raise ValueError('Layer ' + self.name + ' expects ' +
                         str(len(self.states)) + ' states, ' +
                         'but it received ' + str(len(states)) +
                         ' state values. Input received: ' + str(states))
      for index, (value, state) in enumerate(zip(states, self.states)):
        if hasattr(self.cell.state_size, '__len__'):
          dim = self.cell.state_size[index]
        else:
          dim = self.cell.state_size
        if value.shape != get_tuple_shape(dim):
          raise ValueError('State ' + str(index) +
                           ' is incompatible with layer ' +
                           self.name + ': expected shape=' +
                           str(get_tuple_shape(dim)) +
                           ', found shape=' + str(value.shape))
        # TODO(anjalisridhar): consider batch calls to `set_value`.
        K.set_value(state, value)

class StackedSTLSTMCells(Layer):
  """      
  cells: List of RNN cell instances.
  Examples:
  ```python
      cells = [
          STLSTMCell(output_dim),
          STLSTMCell(output_dim),
          STSTMCell(output_dim),
      ]
      inputs = keras.Input((timesteps, input_dim))
      x = keras.layers.RNN(cells)(inputs)
  ```
  """

  def __init__(self, cells, **kwargs):
    if not hasattr(cells, '__len__'):
      raise ValueError('Only a list of STLSTMCells'
                       'is stackable.',
                       'received input:', cells)
    first = True
    for cell in cells:
      if not isinstance(cell, STLSTMCell):
        raise ValueError('All cells must be a `STLSTMCell`'
                         'received cell:', cell)
      if first:
        first = False
        continue

      else:
        cell.state_size = cell.state_size[1:] # remove m-states from non 1st cells.
      
    self.cells = cells
    super(StackedSTLSTMCells, self).__init__(**kwargs)

  @property
  def data_format(self):
    return self.cells[0].data_format

  
  @property
  def state_size(self):
    # States are a flat list of the individual cell state size.
    # e.g. state  s of a 3-layer STLSTM would be '[m, h1, c1, h2, c2, h3, c3]'
    state_size = ()
    for cell in self.cells:
      state_size += cell.state_size
    return state_size

  def call(self, inputs, states, constants=None, **kwargs):
        
    # Recover per-cell states.
    nested_states = []
    first = True
    for cell in self.cells:
      if first:
        nested_states.append(states[:len(cell.state_size)])
        states = states[len(cell.state_size):]
        first = False
      else:
        nested_states.append(states[:len(cell.state_size)])
        states = states[len(cell.state_size):]
    # Call the cells in order and store the returned states.
    new_nested_states = []
    first = True
    for cell, states in zip(self.cells, nested_states):
      if not first:
        states += tuple(m_state)
      
      if generic_utils.has_arg(cell.call, 'constants'):
        inputs, states = cell.call(inputs, states, constants=constants,
                                   **kwargs)
        m_state = [states[-1]]
      else:
        inputs, states = cell.call(inputs, states, **kwargs)
        m_state = [states[-1]]
      new_nested_states.append(states[:-1])
      first = False
   
    # append the last m_state to the first cell's states
    new_nested_states[0].append(states[-1])
    # format the new states as a flat list

    new_states = []
    for cell_states in new_nested_states:
      new_states += cell_states
    
    return inputs, new_states
  
  @property
  def output_size(self):
    return self.cells[-1].output_size

  @tf_utils.shape_type_conversion
  def build(self, input_shape):
    for cell in self.cells:
      cell.build(input_shape)
      output_dim = cell.output_size
      input_shape = input_shape[:2] + tuple(output_dim.as_list())[1:]
    self.built = True

  def get_config(self):
    cells = []
    for cell in self.cells:
      cells.append({
          'class_name': cell.__class__.__name__,
          'config': cell.get_config()
      })
    config = {'cells': cells}
    base_config = super(StackedSTLSTMCells, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  @classmethod
  def from_config(cls, config, custom_objects=None):
    from tensorflow.python.keras.layers import deserialize as deserialize_layer  # pylint: disable=g-import-not-at-top
    cells = []
    for cell_config in config.pop('cells'):
      cells.append(
          deserialize_layer(cell_config, custom_objects=custom_objects))
    return cls(cells, **config)

  @property
  def trainable_weights(self):
    if not self.trainable:
      return []
    weights = []
    for cell in self.cells:
      if isinstance(cell, Layer):
        weights += cell.trainable_weights
    return weights

  @property
  def non_trainable_weights(self):
    weights = []
    for cell in self.cells:
      if isinstance(cell, Layer):
        weights += cell.non_trainable_weights
    if not self.trainable:
      trainable_weights = []
      for cell in self.cells:
        if isinstance(cell, Layer):
          trainable_weights += cell.trainable_weights
      return trainable_weights + weights
    return weights

  def get_weights(self):
    """Retrieves the weights of the model.
    Returns:
        A flat list of Numpy arrays.
    """
    weights = []
    for cell in self.cells:
      if isinstance(cell, Layer):
        weights += cell.weights
    return K.batch_get_value(weights)

  def set_weights(self, weights):
    """Sets the weights of the model.
    Arguments:
        weights: A list of Numpy arrays with shapes and types matching
            the output of `model.get_weights()`.
    """
    tuples = []
    for cell in self.cells:
      if isinstance(cell, Layer):
        num_param = len(cell.weights)
        weights = weights[:num_param]
        for sw, w in zip(cell.weights, weights):
          tuples.append((sw, w))
        weights = weights[num_param:]
    K.batch_set_value(tuples)

  @property
  def losses(self):
    losses = []
    for cell in self.cells:
      if isinstance(cell, Layer):
        losses += cell.losses
    return losses + self._losses

  @property
  def updates(self):
    updates = []
    for cell in self.cells:
      if isinstance(cell, Layer):
        updates += cell.updates
    return updates + self._updates

class STLSTMCell(Layer):
  """Cell class for the Spatio-Temporal LSTM.
  The implementation is based on: https://arxiv.org/pdf/1804.06300.pdf.
  

  # Arguments
      filters: Integer, the dimensionality of the output space
          (i.e., the number of output filters in the convolution operation)
      kernel_size: An integer or tuple/list of n integers, specifying the
          dimensions of the convolution window.
      strides: An integer or tuple/list of n integers,
          specifying the strides of the convolution operation.
          Setting any stride value != 1 is not compatible with setting
          any dialation rate value != 1.
      padding: "valid" or "same".
      data_format: "channels_last" or "channels_first". The default is
          "channels_last" unless 'image_data_format' found in Keras config
          at '~/.keras/keras.json'
      dilation_rate: An integer or tuple/list of n integers, specifying
          the dilation rate to use for dilated convolution operation.
          Currently, setting any dilation rate value != 1 is not compatible
          with setting any strides value != 1
      activation: The activation to be used. Setting no value for this will
          result in the linear activation (i.e., a(x) = x).
      recurrent_activation: Activation for the recurrent steps.
      use_bias: Bool, whether to use a bias vector.
      kernel_initializer: Initialzier for the kernel weights.
      recurrent_initialzier: Initializer for the recurrent kernel weights.

  """

  def __init__(self,
               filters,
               kernel_size,
               strides=(1, 1),
               padding='same',
               data_format=None,
               dilation_rate=(1, 1),
               activation='tanh',
               recurrent_activation='hard_sigmoid',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               recurrent_initializer='orthogonal',
               bias_initializer='zeros',
               unit_forget_bias=True,
               kernel_regularizer=None,
               recurrent_regularizer=None,
               bias_regularizer=None,
               kernel_constraint=None,
               recurrent_constraint=None,
               bias_constraint=None,
               #dropout=0.,
               #recurrent_dropout=0.,
               **kwargs):
    super(STLSTMCell, self).__init__(**kwargs)
    self.filters = filters
    self.kernel_size = conv_utils.normalize_tuple(kernel_size, 2, 'kernel_size')
    self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')
    self.padding = conv_utils.normalize_padding(padding)
    self.data_format = conv_utils.normalize_data_format(data_format)
    self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, 2,
                                                    'dilation_rate')
    self.activation = activations.get(activation)
    self.recurrent_activation = activations.get(recurrent_activation)
    self.use_bias = use_bias

    self.kernel_initializer = initializers.get(kernel_initializer)
    self.recurrent_initializer = initializers.get(recurrent_initializer)
    self.bias_initializer = initializers.get(bias_initializer)
    self.unit_forget_bias = unit_forget_bias

    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
    self.bias_regularizer = regularizers.get(bias_regularizer)

    self.kernel_constraint = constraints.get(kernel_constraint)
    self.recurrent_constraint = constraints.get(recurrent_constraint)
    self.bias_constraint = constraints.get(bias_constraint)

#    self.dropout = min(1., max(0., dropout))
#    self.recurrent_dropout = min(1., max(0., recurrent_dropout))
    self.dropout = 0.
    self.recurrent_dropout = 0.
    self._state_size = (self.filters, self.filters, self.filters)
    self._dropout_mask = None
    self._recurrent_dropout_mask = None 

  @property
  def state_size(self):
    return self._state_size
  
  @state_size.setter
  def state_size(self, state_tuple):
    if type(state_tuple) is not tuple:
      raise ValueError('The state shuoud be tuple of state sizes')
    self._state_size = state_tuple

  
  def build(self, input_shape, **kwargs):

    if self.data_format == 'channels_first':
      channel_axis = 1
      space = input_shape[2:]
      new_space = []
      for i in range(len(space)):
        new_dim = conv_utils.conv_output_length(
            space[i],
            self.kernel_size[i],
            padding=self.padding,
            stride=self.strides[i],
            dilation=self.dilation_rate[i])
        new_space.append(new_dim)
      self.output_size = tensor_shape.TensorShape([input_shape[0], self.filters] + new_space)

    else:
      channel_axis = -1
      space = input_shape[2:-1]
      new_space = []
      for i in range(len(space)):
        new_dim = conv_utils.conv_output_length(
            space[i],
            self.kernel_size[i],
            padding=self.padding,
            stride=self.strides[i],
            dilation=self.dilation_rate[i])
        new_space.append(new_dim)
      self.output_size = tensor_shape.TensorShape([input_shape[0]] + new_space + [self.filters])

    if input_shape[channel_axis] is None:
      raise ValueError('The channel dimension of the inputs '
                       'should be defined. Found `None`.')
    input_dim = input_shape[channel_axis]
    
    
    shape_x = self.kernel_size + (input_dim, self.filters * 7)
    shape_m = self.kernel_size + (self.filters, self.filters * 4)
    shape_h = self.kernel_size + (self.filters, self.filters * 4)
    shape_c = self.kernel_size + (self.filters, self.filters)
    shape_1by1 = conv_utils.normalize_tuple(1, 2, 'kernel_size') +\
        (self.filters * 2, self.filters)
    self.kernel_shape = shape_x
    
    self.kernel_x = self.add_weight(shape=shape_x,
                                    initializer=self.kernel_initializer,
                                    name='kernel_x',
                                    regularizer=self.kernel_regularizer,
                                    constraint=self.kernel_constraint)
    self.kernel_m = self.add_weight(shape=shape_m,
                                    initializer=self.recurrent_initializer,
                                    name='kernel_m',
                                    regularizer=self.recurrent_regularizer,
                                    constraint=self.recurrent_constraint)
    self.kernel_h = self.add_weight(shape=shape_h,
                                    initializer=self.recurrent_initializer,
                                    name='kernel_h',
                                    regularizer=self.recurrent_regularizer,
                                    constraint=self.recurrent_constraint)
    self.kernel_c = self.add_weight(shape=shape_c,
                                    initializer=self.recurrent_initializer,
                                    name='kernel_c',
                                    regularizer=self.recurrent_regularizer,
                                    constraint=self.recurrent_constraint)
    self.kernel_1by1 = self.add_weight(shape=shape_1by1,
                                       initializer=self.kernel_initializer,
                                       name='kernel_1by1',
                                       regularizer=self.kernel_regularizer,
                                       constraint=self.kernel_constraint)


    if self.use_bias:
      if self.unit_forget_bias:

        def bias_initializer(_, *args, **kwargs):
          return K.concatenate([
              self.bias_initializer((self.filters,), *args, **kwargs),
              initializers.Ones()((self.filters,), *args, **kwargs),
              self.bias_initializer((self.filters * 2,), *args, **kwargs),
              initializers.Ones()((self.filters,), *args, **kwargs),
              self.bias_initializer((self.filters * 2,), *args, **kwargs)
          ])
      else:
        bias_initializer = self.bias_initializer
      self.bias = self.add_weight(
          shape=(self.filters * 7,),
          name='bias',
          initializer=bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint)

    else:
      self.bias = None

    self.kernel_xi = self.kernel_x[:, :, :, : self.filters]
    self.kernel_xf = self.kernel_x[:, :, :, self.filters: self.filters * 2]
    self.kernel_xc = self.kernel_x[:, :, :, self.filters * 2: self.filters * 3]
    
    self.kernel_hi = self.kernel_h[:, :, :, : self.filters]
    self.kernel_hf = self.kernel_h[:, :, :, self.filters: self.filters * 2]
    self.kernel_hc = self.kernel_h[:, :, :, self.filters * 2: self.filters * 3]

    self.kernel_xip = self.kernel_x[:, :, :, self.filters * 3: self.filters * 4]
    self.kernel_xfp = self.kernel_x[:, :, :, self.filters * 4: self.filters * 5]   
    self.kernel_xm = self.kernel_x[:, :, :, self.filters * 5: self.filters * 6]

    self.kernel_mi = self.kernel_m[:, :, :, : self.filters]
    self.kernel_mf = self.kernel_m[:, :, :, self.filters: self.filters * 2]
    self.kernel_mm = self.kernel_m[:, :, :, self.filters * 2: self.filters * 3]

    self.kernel_xo = self.kernel_x[:, :, :, self.filters * 6:]
    self.kernel_ho = self.kernel_h[:, :, :, self.filters * 3:]
    self.kernel_co = self.kernel_c
    self.kernel_mo = self.kernel_m[:, :, :, self.filters * 3:]

    if self.use_bias:
      self.bias_i = self.bias[:self.filters]
      self.bias_f = self.bias[self.filters: self.filters * 2]
      self.bias_c = self.bias[self.filters * 2: self.filters * 3]
      self.bias_ip = self.bias[self.filters * 3: self.filters * 4]
      self.bias_fp = self.bias[self.filters * 4: self.filters * 5]
      self.bias_m = self.bias[self.filters * 5: self.filters * 6]
      self.bias_o = self.bias[self.filters * 6:]
    else:
      self.bias_i = None
      self.bias_f = None
      self.bias_c = None
      self.bias_ip = None
      self.bias_fp = None
      self.bias_m = None
      self.bias_o = None
    
    self.built = True

  def call(self, inputs, states, training=None):
#    if 0 < self.dropout < 1 and self._dropout_mask is None:
#      self._dropout_mask = _generate_dropout_mask(
#          K.ones_like(inputs),
#          self.dropout,
#          training=training,
#          count=4)
#    if (0 < self.recurrent_dropout < 1 and
#        self._recurrent_dropout_mask is None):
#      self._recurrent_dropout_mask = _generate_dropout_mask(
#          K.ones_like(states[1]),
#          self.recurrent_dropout,
#          training=training,
#          count=4)
#
#    # dropout matrices for input units
#    dp_mask = self._dropout_mask
#    # dropout matrices for recurrent units
#    rec_dp_mask = self._recurrent_dropout_mask
#
    h_tm1 = states[0]  # previous memory state
    c_tm1 = states[1]  # previous carry state
    m_tm1 = states[2]  # zigzag state

    inputs_i = inputs
    inputs_f = inputs
    inputs_c = inputs
    inputs_ip = inputs
    inputs_fp = inputs
    inputs_m = inputs
    inputs_o = inputs

    h_tm1_i = h_tm1
    h_tm1_f = h_tm1 
    h_tm1_c = h_tm1
    h_tm1_o = h_tm1

#
#    if 0 < self.dropout < 1.:
#      inputs_i = inputs * dp_mask[0]
#      inputs_f = inputs * dp_mask[1]
#      inputs_c = inputs * dp_mask[2]
#      inputs_o = inputs * dp_mask[3]
#    else:
#      inputs_i = inputs
#      inputs_f = inputs
#      inputs_c = inputs
#      inputs_o = inputs
#
#    if 0 < self.recurrent_dropout < 1.:
#      h_tm1_i = h_tm1 * rec_dp_mask[0]
#      h_tm1_f = h_tm1 * rec_dp_mask[1]
#      h_tm1_c = h_tm1 * rec_dp_mask[2]
#      h_tm1_o = h_tm1 * rec_dp_mask[3]
#    else:
#      h_tm1_i = h_tm1
#      h_tm1_f = h_tm1
#      h_tm1_c = h_tm1
#      h_tm1_o = h_tm1
    

    x_i = self.input_conv(inputs_i, self.kernel_xi, self.bias_i,
                          padding=self.padding)
    x_f = self.input_conv(inputs_f, self.kernel_xf, self.bias_f,
                          padding=self.padding)
    x_c = self.input_conv(inputs_c, self.kernel_xc, self.bias_c,
                          padding=self.padding)
    x_ip = self.input_conv(inputs_ip, self.kernel_xip, self.bias_ip,
                           padding=self.padding)
    x_fp = self.input_conv(inputs_fp, self.kernel_xfp, self.bias_fp,
                           padding=self.padding)
    x_m = self.input_conv(inputs_m, self.kernel_xm, self.bias_m,
                          padding=self.padding)
    x_o = self.input_conv(inputs_o, self.kernel_xo, self.bias_o,
                          padding=self.padding)

    h_i = self.recurrent_conv(h_tm1_i,
                              self.kernel_hi)
    h_f = self.recurrent_conv(h_tm1_f,
                              self.kernel_hf)
    h_c = self.recurrent_conv(h_tm1_c,
                              self.kernel_hc)
    h_o = self.recurrent_conv(h_tm1_o,
                              self.kernel_ho)
    m_i = self.recurrent_conv(m_tm1,
                              self.kernel_mi)
    m_f = self.recurrent_conv(m_tm1,
                              self.kernel_mf)
    m_m = self.recurrent_conv(m_tm1,
                              self.kernel_mm)
    i = self.recurrent_activation(x_i + h_i)
    f = self.recurrent_activation(x_f + h_f)
    c = f * c_tm1 + i * self.activation(x_c + h_c)
    c_o = self.recurrent_conv(c,
                              self.kernel_co)

    ip = self.recurrent_activation(x_ip + m_i)
    fp = self.recurrent_activation(x_fp + m_f)
    m = fp * m_tm1 + ip * self.activation(x_m + m_m)
    m_o = self.recurrent_conv(m,
                              self.kernel_mo)

    cm_h = self.recurrent_conv(K.concatenate([c, m]),
                               self.kernel_1by1)


    o = self.recurrent_activation(x_o + h_o + c_o + m_o)
    h = o * self.activation(cm_h)

    return h, [h, c, m]

  def input_conv(self, x, w, b=None, padding='same'):
    conv_out = K.conv2d(x, w, strides=self.strides,
                        padding=padding,
                        data_format=self.data_format,
                        dilation_rate=self.dilation_rate)
    if b is not None:
      conv_out = K.bias_add(conv_out, b,
                            data_format=self.data_format)
    return conv_out

  def recurrent_conv(self, x, w):
    conv_out = K.conv2d(x, w, strides=(1, 1),
                        padding='same',
                        data_format=self.data_format)
    return conv_out

  def get_config(self):
    config = {'filters': self.filters,
              'kernel_size': self.kernel_size,
              'strides': self.strides,
              'padding': self.padding,
              'data_format': self.data_format,
              'dilation_rate': self.dilation_rate,
              'activation': activations.serialize(self.activation),
              'recurrent_activation': activations.serialize(
                  self.recurrent_activation),
              'use_bias': self.use_bias,
              'kernel_initializer': initializers.serialize(
                  self.kernel_initializer),
              'recurrent_initializer': initializers.serialize(
                  self.recurrent_initializer),
              'bias_initializer': initializers.serialize(self.bias_initializer),
              'unit_forget_bias': self.unit_forget_bias,
              'kernel_regularizer': regularizers.serialize(
                  self.kernel_regularizer),
              'recurrent_regularizer': regularizers.serialize(
                  self.recurrent_regularizer),
              'bias_regularizer': regularizers.serialize(self.bias_regularizer),
              'kernel_constraint': constraints.serialize(
                  self.kernel_constraint),
              'recurrent_constraint': constraints.serialize(
                  self.recurrent_constraint),
              'bias_constraint': constraints.serialize(self.bias_constraint)}
              #'dropout': self.dropout,
              #'recurrent_dropout': self.recurrent_dropout}
    base_config = super(STLSTMCell, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

