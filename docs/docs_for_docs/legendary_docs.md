# Metabox docs 写作指引

OK 这就是传说中的文档的文档，写这个文档主要为了帮助师兄/后来的学弟对于 Metabox 文档维护和 readthedocs 快速上手。

没事。不会再有文档的文档的文档的。

## 前言

readdocs 用 sphinx 主要构成，Sphinx 是 Python 社区编写和使用的文档构建工具，除了天然支持 Python 项目以外，Sphinx 对 C/C++ 项目也有很好的支持，并在不断增加对其它开发语言的支持。

- [Sphinx 网站](http://sphinx-doc.org/)
- [Sphinx 使用手册](https://zh-sphinx-doc.readthedocs.io/en/latest/index.html)
- [非常好的教程](https://sphinx-practise.readthedocs.io/zh-cn/latest/index.html)

Sphinx 使用 reST 作为标记语言，其与 markdown 非常相似。

- [reStructuredText 网站](http://docutils.sf.net/rst.html)

快速上手：

- [Read the Docs 从懵逼到入门](https://blog.csdn.net/lu_embedded/article/details/109006380)

对着写了一个 readthedocs 的最小实现，有时间可以看看。（也可以顺便 star）虽然很水）

- [learnreadthedocs](https://github.com/hohq/learn_readdocs)

> 当然，以上所有都是 当你完全懵逼的好的查询资料，就算不看应该也没啥问题））

那接下来就开始吧。（我是边写边记录的，如有缺漏请谅解）

## 环境(或者我们写个 requirement.txt)

请运行

```shell
pip3 install -U Sphinx

pip3 install sphinx-autobuild
pip3 install sphinx_rtd_theme
pip3 install recommonmark
pip3 install sphinx_markdown_tables
pip install myst-parser
```

## 结构介绍

### conf.py

这是 sphinx 进行构建的关键文件，不过我们使用倒是不用太深究。
主要有几个点是可以注意的：

- extension(扩展)会在其中进行添加
- 主题可以在其中进行调整，这里的主题一般是 readthedocs/sphinx 自带的，
- 界面是可以通过使用 css 进一步美化的，也是在这里进行导入
- 扩展的设置也在这里做

### 主界面

跟一般的前端 web 界面类似，我们的主界面是在 Metabox/docs/source/index.md, 如果要修改主界面请在这里。

## markdown 与 readthedocs 的一些使用问题

感觉上 readthedocs 和 reST 语法更合得来（毕竟是人家的默认语言）
但使用 markdown 也不是不行，而且这样也可以节约学习成本）

### 指令

请使用

> \```{指令}
>
> \```
>
> 的方式进行调用

e.g.(tocree 为 sphinx 的指令)

````markdown
```{toctree}
:caption: 'API Reference'
:maxdepth: 1
:hidden:

apidocs/index
```
````

### 链接

这个提一嘴
或许需要跳转页面,那你就按照 markdown 的链接语法写:

```markdown
[title](link)
```

如果为内部文档链接,请使用相对路径
e.g.

```markdown
1. [QuickStart](guide/QuickStart/index)
```

## 进行测试

请在 Metabox/docs 目录下
运行

```shell
make html
```

以来构建项目

构建完成后，使用 sphinx-autobuild 进行快速 build

```shell
sphinx-autobuild source build/html
```

项目将映射到 [http://127.0.0.1:8000](http://127.0.0.1:8000) 也就是本地 8000 端口，访问即可。
项目应该是动态更新的。
有问题看 sphinx 文档吧，前有链接。

> 请不要把生成的 build 文件夹上传)))
> P.S.可能有段时间我们或许会遇到路径问题(在 api reference autodocs 时)，不过慢慢来吧。

## 本地化相关/多语言相关

evox 使用的是运用大语言模型进行翻译，我们直接把 translate 的脚本扒过来即插即用是可以的。（已试验）
只不过我手上是没有 openai 的 apikey 的，没办法直接用，试用了智谱清言但是好像不太聪明。
如果用 llm 来翻译总时长可能在 1h 以上，我没跑完））

还是先记录一下怎么进行多语言

### 生成多语言版本的文件

参考：[Sphinx+Read the Docs 的多语言版本文档实现](https://zhuanlan.zhihu.com/p/427843476)
不用看 Transifex 了，那玩意应该是线上人工协作的。

安装环境

```shell
pip install sphinx-intl
```

```shell
sphinx-build -b gettext ./source build/gettext
```

此时，docs/build/gettext 中会生成原来文档 rst 文件对应的.pot 文件。

我们要为对应的语言的文档生成对应的 pot 文件：

```shell
sphinx-intl update -p ./build/gettext -l zh_CN
```

我们使用在 docs/locales 的目录下的那坨文件就行

翻译完后，用一下指令构建：

```shell
sphinx-build -b html -D language=zh_CN ./source/ build/html/zh_CN
```

sphinx-build 将把 po 文件构建为 mo 文件，并且你可以在 build 里看见构建好的 web 文件
但是似乎不能用 autobuild 自动映射了（因为他会重新根据默认语言构建一遍，然后你就看不到了），不过应该是小问题，最终效果还是有的。

![中文版](docs/docs_for_docs/image/legendary_docs/image.png)

# 记录

RL 用 ppo 为例子，算法来源，函数接口，函数的全称

# 问题记录：

1. Q：代码块没有复制小图标 A：使用 sphinx_copybutton 插件，完成（应该可以自定义样式？[link](https://sphinx-copybutton.readthedocs.io/en/latest/index.html)）
2. Q：不支持!NOTE A:下面是解决方案，因为我们使用的是 mystparseer，根据[官方文档](https://myst-parser.readthedocs.io/en/latest/intro.html)
   我们可以这样使用：
<!-- 
```

```{note} Notes require **no** arguments, so content can start here.
```
```{tip} Notes require **no** arguments, so content can start here.
```
```{warning} Notes require **no** arguments, so content can start here.
```
:::{note}
This text is **standard** _Markdown_
:::
```{admonition} Here's my title
:class: note

Here's my admonition content

```

``` -->

这语法好麻烦。。。。注释了自己看吧。。。。


