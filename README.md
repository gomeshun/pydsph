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
  
  バグが起きそうな原因というのは、手でパラメータの順番を陽に・陰に
  指定してしまうこと。これをなくすには、辞書でパラメータを指定するようにすればいい。
  ただし、Likelihoodの評価の時にだけは順序が関係してくる。
  そのため、Likelihoodの引数の順序は特定の規則に従って勝手に決まるようにして、順番が現れないようにすればよい。
  
  あれ、ていうか、よく考えるとパラメータに値をAssignする必要はないような？？
  なぜなら、なんかモデルを評価するときにモデルにパラメータをassignしているわけではない...
  
  なにかモデルを評価するときには、パラメータ空間上の特定の点に対してLikelihoodの値を評価する、ということをやる。その意味では、パラメータはむしろ一般のclassとして定義すべきであるきがする。その意味だと、Modelはパラメータ空間を持ち、パラメータ空間の特定の一点でモデルを評価する、ということになる。クラスがクラスを持つ、とはどういうことか？？？
  
  今まで考えていたのはむしろ「パラメータはspecified/unspecifiedという属性を持つ」ということ。こっちのがわかりやすい気がする。
  で、モデルの評価について考えると、
  モデルはパラメータを持つ。
  モデルは、SpecifiedなParameterを持っているとき、Likelihoodの評価ができる。
      モデルがSpecifiedされているとは、モデルのパラメータがすべてSpecifiedされてることである。
      モデルは、複モデルを持つ。
      モデルがSpecifiedされているとき、複モデルもすべてSpecifiedである。
      モデルがSpecifiedされているとは、モデルパラメータが値を持つことである。
      モデルが値を持つとき、複モデルも値を持つ。
  モデルを評価するとは、
      Specifiedなパラメータの値に基づき、Likelihood(values)を求めること。
      Specifiedされていないときは、エラーになる。
      
    あるいは、関数の評価をするときには必ず値をassignするようにして、関数内部ではself.parameter["name_of_parmaeter"]という風に呼ぶようにするのでいいかもしれない。
    これだと、どの関数がどの変数に依存しているのかわかりづらいという欠点があるが…
    
    変数を明示するには、変数を引数に書いとくのが良い。
    そこで、
    func_test(x,a,b,c)
    みたいな関数を定義したとき、自動で
    test(x,p)
    みたいな関数を再定義してくれるようにしたい。        
        param_names = ["a","b"]
        
        def func_test(x,b,a):
    これならわざわざ最初に変数を自分で定義しなおす必要はない。
    この関数は上のレイヤーのモデルから簡単に呼び出せる。
    
    。。。いや、待てよ？関数の引数は必ずしも全部のパラメータを尽くすわけではないから、
    都合よく変数を抽出する必要がある。これはちょっとめんどくさい…
    …やっぱり、最初にモデルの変数をAssignするようにしてしまおう。
    
    ...しかしこれも問題がある。最初にモデルのパラメータ全部をAssignしてしまうと、
    下のモデルの関数を呼び出すときに再度Assignが発生する気がする…
    
    これは使う関数を分ければよい。モデルの定義に使う関数と、実際の評価に使う関数を分ける:
    
        定義に使う関数　`func_f`:
            ```python:
            def func_f(self,x,a,b,Parameter_Model1,Parameter_Model2):
                ...
                self.Model1.func_f1(x,*Paramster_Model1)
            ...
                
        呼び出し方　f(x):この時、パラメータは事前にAssignされる。
        
     …これもなんか問題があるな…
     例えばLikelihoodとかは（例えばというかこれが主目的なのだが）
     likelihood(*p) ないし likelhood(p)
     とかいう形で呼び出す（ようなものを定義しないとMCMCで使えない）。
     ここでpはFlatな配列になっている。
     このLikelihoodの定義中で呼び出すモデルの関数は、pを引数にとらないといけない。
     （あるいは、Likelihoodの評価前に事前にAssignを行っておく。）
     定義を書くときには、モデルの関数のP依存性は…書きたい？？
     よく考えると、一個下のモデルの関数を何かしらで呼び出すことはある。
     たとえば、Likelihoodはdsphのモデルの関数sigmalosを呼び出すので、
     こいつにはどうにかしてpの情報のうちdsphモデルに相当する分をあげないといけない。
     
     方針としては、パラメータを引数に持つような関数は事前にモデルにAssignをしてから使うようにする、というのがあげられるが、これだとモデルを上に登って行ったとき二重Assignの問題が常にある。
     
     ちゃんと考えるために、良さげな候補となる関数の書き方をいくつかにグループ分けする：
     
     1. func_raw(x,y,a,b,Parameter_Model1,Parameter_Model2)
     1. func_flat(x,p)
     1. func_using_assigned_parameter(x)
         
     それぞれ、内部をどういう風に記述したらいいか考えてみると、
     
     ```
     def func_raw(self,x,y,a,b,Parameter_Model1,Parameter_Model2):
         (... some calculation with a,b... )
         func_Model1_flat(x,Parameter_Model1)
     ```
     
     ```
     def func_flat(self,x,p):  # likelihoodはどうにかしてこの形式のものを用意する必要がある
         a,b,Parameter_Model1,Parameter_Model2 = self.split_param(p)
         (... some calculation with a,b... )
         func_Model1_flat(x,Parameter_Model1)
         
     # Usage:
     func_flat(x,p)
     ```
     
     ```
     def func_using_assigned_parameter(x):
         a,b = self.parameter[["a","b"]]  # あるいは直接これを式中で呼び出す
         func_Model1_using_assigned_parameter(x)  # assignされている値を使う
         func_Model2_flat(y,self.parameter["Parameter_Model2"])　# 代入をする
         
     # Usage:
     (some assignment procedure before calling)
     func_using_assigned_parameter(x)
     ```
     
     それぞれメリット・デメリットがある。
     
     1. 
        - ! 変数が明確。
        - ? Likelihoodの形とずれてるので、少しだけ変更が必要。
        - ? 上下のモデル全部含めて全部raw形式ではかけない（内部でflat形式を呼んでいるので）
     1. 
        - ! Likelihoodの形とあっているので、そのまま使える。
        - ? 内部でパラメータの展開が必要。
     1. 
        - ! いちいち内部で下のモデルのパラメータを気にする必要がない。
        - ? 実行前に事前にAssignが必要。
        - ? 関数が何に依存してるのかがパット見でいまいちわかりにくい。
           - てかこれって結局最初に自分のパラメータだけは展開がいるんじゃね？
           
     それぞれの関数はwrapperを作ることで行き来ができるから、
     一番**表記上**望ましいものを採用するべき。
     
     表記上望ましいのはraw形式なので、これを採用することにする。
     rawからflatへの行き来は、ラッパを作ることにする。
     
     func_と定義された関数は、自動的にflat形式の関数に展開することにする。
     どうするか…
     
     flat な変数を受けてrawの変数に展開するには、
     一旦モデルのパラメータとしてAsssignしているとこだけ取り出すか、
     ダイレクトに配列をそのまま分けるやり方がある。
     
     モデルのパラメータとしてAssignするのは…？
     もし何かモデルがパラメータの実現値を持っているなら合理的な気がするけど、
     まだ値がわからん段階で何か値をAssignするというのは違う気がする。
     
     モデルのパラメータじゃなくてAssignするのはよさそう。
     つまり一時的にパラメータのインスタンスを作成して、
     そこに値をAssignし、そこから値を読み出す。
     …めんどくせえなこれ…これってしかもわざわざインスタンスを生成しているので遅そう。
     （一方で実際にモデルが何かの値を持っているということもありそうだから、
     これはこれとして別に実装するのがよさそう。）
     
     一方、ダイレクトに分けるのは簡単で、既に実装済みである。
     （Assignのやり方と比較して気になるのは、パラメータを分割するときのコスト。
     そんなでもないか？？？→そんなでもなさそう。）
     単にLikelihoodを評価するだけなら、これでよい。
     
     クラスメソッドとして定義するのが良いであろう。
     なぜなら別に特定のインスタンスによっているわけではないから。
     ただしStaticmethodだとおかしなことになる（submodelによるので）。
     
     どの関数を直して、どの関数を直さないかはどうやって判別するか。
     デコレータを使ってしまうのが良いだろう。
     
     ```python:
     @deco
     func(cls,x,a,b,c):
        pass
     ```
     
     とかやると、勝手にこの関数を変換し、クラスメソッドとして登録する用にしたい。
     問題なのは、変更した後の名前。func -> func_flatten でいいか。
     …いいか？冗長でない？
     むしろ変更する前の関数を何か別の名前にするか…どっちがいいかな…
     呼ばれ方を上のリストを元に考えると、
     元の関数を `_func` としてしまい、新しい関数を`func`にするのが良さそう。
     
     そのためには…
     
     デコレータめんどくさそうなので、従来の方式で行く（astropyもこういう理由なのか？）。
     
     やりたいことは、
     
     - クラスから呼び出されたら、与えられた値を使って評価する。
     
     
     デコレータにするためには、どうしてもパラメータの情報がいる。頭に渡すことにしてもいいが：
     
     ```python
     @deco(parameter)
     ```
     
     冗長でない？？？
     そうでもないか。これでどのパラメータがpackされるかがわかりやすくなるかな？
     自分以外のパラメータをPackしてしまう危険もあるが…まあ大丈夫だろう。
     後は、継承とかしたときに大丈夫か…
     継承してパラメータを増やしたりされた時、デコレータがおかしくならないか？
     ともかくやってみる。
     
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
                - during the calculation, parameters of Model 
    
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
    - `Parameter` has `sub_params`.
        - `sub_params` is a list of `Parameter`s.
    - `Parameter` can `clear_value`.
    - `Parameter` can `assign_value`.
    - * `Parameter` can `convert_unit`.
    - `Parameter` has `belonged_model`.
    - `Parameter` has `dim`.
    - `Parameter` can `__len__``
    - `Parameter` can `__or__` or `|`.
    - `Parameter` can `add` other parameters into `self.sub_params`.

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

# Note

Some usual frameworks:

- A `Dog` has a `tail`.
    - An instance of `Dog` class has an attribute `tail`
- The `Dog` has a `binomen`.
    - The `Dog` class has an attribute `binomen`.
- A `Dog` is an `Animal`.
    - An instance of the `Dog` class is also an instance of the `Animal` class.

