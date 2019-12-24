# MemFG
**MemFG** is a module written in Python, which allows us to

- define user-custom models of dwarf spheroidal galaxies (dSphs),
- evaluate, test, and compare dSph models,
- search the parameter region of a given dSph,
- estimate J-factor PDFs or confidencial intervals.

**MemFG** module is implemented to satisfy that

- dSph models can be defined by using a specific model format in MemFG,
- all of the output files yielded by MemFG have a user-friendly and easily-reuseable format,
- statistical frameworks can be swicthed between such as frequentist statistics or Bayesian statistics.
- sampling algorithm can be switched among several MCMC or nested samplers.




# Todo

## `Parameter` class? 

- Usually, a likelihood function has many parameters (more than 10).
  Let us consider that we pass these parameters to the likelihood as:
  
  ```
  likelihood(p0,p1,p2,p3,p4,p5,...)
  ```
  
  This may cause some bugs such that
  
  ```
  likelihood(p0,!p2,!p3,!p1,p4,...)
  ```
  
  Of cource, we can avoid such bugs by taking very careful to the order of the arguments.
  However, if we can just call the likelihood as:
  
  ```
  likelihood(ps)
  ```
  
  this helps us to coding very quickly.
  In order to achieve this feature, we should define a format of the order of the arguments `ps`.
  We should take care that the `emcee.EnsembleSampler` is defined as:
  
  
  > `emcee.EnsembleSampler(nwalkers, dim, lnpostfn, a=2.0, args=[], kwargs={},...)`
  
  where
  
  > `lnpostfn`: A function that takes a **vector** (**numpy.array**) in the parameter space as input and returns the natural logarithm of the posterior probability for that position.
  
  Moreover, we should also take care that the computational cost of the likelihood function should be small.
  
  一番実装が簡単なのは、Arrayをそのまま使ってしまうこと。例えばLikelihoodの中で、
  p0,p1,p2, ...  = ps
  とか展開して、それぞれのpiはこういう意味だから...とか考えながら、適宜Likelihoodの中で使う。
  しかしこれは非常に危ない（取り違えが起きる可能性ありまくる）。
  順番がめちゃめちゃにならないようにどっかでパラメータの順番を明示的に指定してあげて、
  「この変数はこのパラメータ、この変数はこのパラメータ」と教えてあげたいのだが、
  それをやると結局そこに問題を押し付けただけになる（そこで間違えたらどうする？）。
  例えば間違えないように、
  velocity = p["velocity"],
  position = p["position"],
  みたいな代入をしてもいいけれど、これだとなんか関数を定義するたびに頭で毎回こういうことをしないといけなくなる。
  これはいちいち時間を食うし（dict型ならそんなでもない）、冗長である（これはそう）。
  かといって、関数の引数にデデンと全部書いてしまうと、それこそ20個近い変数を全部書くことになって、可視性が悪い。
  
  Likelihood、というか継承・複合しまくったモデルのパラメータが肥大する理由は、各々のモデルの持っているパラメータが全部一つにまとまってしまうから。
  しかし実際はパラメータは各々のモデルにBoundされた概念である。なので、それごとに分けてしまえばそれほど数は大きくならない（はず）。
  
  つまり、psをモデルごとに分けたいので、
  ps_stellar,ps_dm = self.split_params(ps,[stellar_model,dm_model])
  みたいなことをしたい。これならそんなに冗長でもない。
  問題は最下層のモデルでどう展開するか。
  これは、
  a,b,g = self.split_params(p0,["a","b","z"])
  みたいにすれば良いかな？ほんとに？
  これってなんか辞書でアクセスしてるのと似てるから、
  a,b,g = self.params(["a","b","h"])
  みたくすればよいか。
  いやでも、なんか自分でモデルを定義したくなった時にこういう記法を強制させるのはキモい。
  一番最下層のモデルでは、単に何かの関数は単純に
  def func(a,b,g):
      ...
      return ret
  みたく書きたい。しかしこれだと関数の順番がここで決まってしまう...
  一般にモデルにはパラメータが合って、順番が定義されている。その順番で使うように強制したいのだけど。。。
  やはり、最下層のモデルでも
  a,b,g = self.split_params(p0,["a","b","z"])
  みたいな書き方をさせよう。あるいは、デコレータを定義して、ナマの形の関数を変更しよう。
  
  Hence, in order to achieve this feature, we separate the definition of likelihood from the implementation of likelihood.
  In a definition step, 
  
  ```
  def likelihood(Parameters params):
      (some computation)
      return ret
  ```
  
  Then, Model class understand this likelihood function as 
  
  ```
  def likelihood(np.array ps)
      ...
  ```
  
  Is is possible???

## Implementation

To achieve the demands mentioned above, MemFG has the following classes:

- class `Model`: define all physical or statistical models in MemFG.
    - `Model` has `parameters`.
        - `parameters` is a `list` of `Parameter`s.
    - `Model` has `submodels`.
        - `submodels` is a `list` of `Model`s.
        
    - `Model` has `param_names`.
        - `param_names` is a list of parameter.name for parameter in parameters.
        
    - `Model` has `all_parameters`.
        `all_parameters` is a list of `Parameter`s which a `Model` of submodels` has.
        
    - `Model` has some `function`s.
        - `function` has a `body`.
        - `function` has a `args`.
            - `args` is some `names` of `parameters`.
        - `function` can `check_args`: `check_args(params)`:
            - params: a instance of `Parameters`
        - `function` can `__call__`: `__call__(params)`:
            - params: a instance of `Parameters`
    
    - `Model` can `split_params`: `split_params(params)`
    

- * class `Parameters`: ??? (Is it actually required?)
    - `Parameters` is a `list`.　??? (or has a `list` or `dict` of `Parameters`???)
    - `Parameters` has a `names`.
    - `Parameters` has a `values`.
    - `Parameters` can `assign_values`.
    - `Parameters` can `clear_values`.
    
    --> a set of parameters is also parameter !!!
    
- class `Parameter`:
    - `Parameter` has `name`.
        - `name` is a `string` or a `list` of `string`
    - * `Parameter` has `unit`.
        - `unit` is a `string` or a `list` of `string`
    - `Parameter` has `value`.
        - `value` is a `double` or a `list` of `double`
    - `Parameter` can `clear_value`.
    - `Parameter` can `assign_value`.
    - * `Parameter` can `convert_unit`.
    - `Parameter` has `belonged_model`.
    - `Parameter` has `dim`.
    - `Parameter` can `__len__``
    - `Parameter` can `__or__` or `|`.

- class `DMModel`: define dark matter density profile.
    - `DMModel` is a `Model`.
    - `DMModel` has `enclosure_mass`.
    - `DMModel` has `density_profile`.
    
- class `StellarModel`: define stellar models of dSPhs.
    - `StellarModel` is a `Model`.
    - `StellarModel` has 

- class `PlummerModel`: define the Plummer model.
    - `PlummerModel` has a `list` of `Parameter`: `r_half`

- class `dSphModel`: define dSph models.
    - `dSphModel` is a `Model`.
    - `dSphModel` has `stellar_model` and `dm_model`.
        - `stellar_model` is a `StellarModel`.
        - `dm_model` is a `DMModel`.
    - `Model` can `fit` `ObservedData`,
    - `Model` can `evaluate_loglikelihood` based on `ObservedData`,
- ``
- 
- 

# 