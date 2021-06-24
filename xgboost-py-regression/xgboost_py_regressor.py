#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/sigopt/sigopt-examples/blob/master/xgboost_py_regressor.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# ![Two-Sigma_SigOpt_Logo_300px.png](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAASwAAAEsCAYAAAB5fY51AAAYYElEQVR4Ae3Be7zn9YA/8Od5z3fuzcV0m5JpogtKa3NJ2IqIFIWy2RCKZCMZkiRFbqlMQiEtQlqXim5ri5a0lhBKkm50nWouzf1yzvk9fo/+mP3uzJk5l+85n+/nfF/PZxf09vaKiGh3RURETRQRETVRRETURBERURNFRERNFBERNVFERNREERFRE0VERE0UERE1UURE1EQREVETDTV06tyLTrr4Rz9/y9RNJi4VEQPVtfCxpRM+e/KRx+A/9nvRs9VFQw3Ne3Th7Nvvuu/J06dtIiIGbv6CxyxeunxTNdNQQ40xY1aNGzfWuLENETFw48aNVUpZo2aKiIiaKCIiaqKIiKiJIiKiJoqIiJooIiJqooiIqIkiIqImioiImigiImqiiIioiSIioiaKiIiaKCIiaqKIiKiJIiKiJoqIiJooIiJqooiIqIkiIqImioiImigiImqiiIioiSIioiaKiIiaKCIiaqKIiKiJIiKiJoqIiJooIiJqoiGGw+14j+hU3XgC5mJL0TINMRy2wSR8T3SqqfiEaKkihsNEnIytRKcajy7RUkUMl2fgHSKiZYoYTsfg6SKiJYoYTjNwsohoiSKG22vxKhExZEUMtwZOxFQRMSRFjITn4q0iYkiKGAldeD9miYhBK2KkbI0TRcSgFTGSDsdeImJQihhJE3ASxouIAStipL0ErxcRA1ZEFT6IzUXEgBRRhR0xR0QMSBFVORrPFBH9VkRVpuJEjBUR/VJElQ7GASKiX4qoUhdOwjQRsVFFVG03vF1EbFQR7eB92EFEbFBDtIMtcALejm7VGodXYDy6jW7j8Ef8UdRCQ7SLf8ZFuEa1uvFavEFnuBV74hHR9opoF5PxIUxUrW58BPN0hqfhPaIWimgne+MNqncnTtc5jsGuou0V0W5OxBNV73zcqDNMw4kYI9paEe1mNuao3iJ8At06wyF4hWhrRbSjN2EP1bsEl+kMBSdhE9G2imhHm+J4jFW9U7FQZ3gu3ibaVhHt6iAcpHp/wBd1juMxW7SlItrZRzBd9b6IP+kMM/EBjBFtp4h2tjOOVb378Fn06AyH4YWi7RTR7o7CLqp3Af5LZ5iCEzFetJUi2t1WeD+6VKsHJ2O5zrAvDhVtpYg6eD1eonq/wNd1jpOwpWgbRdTBWHwU41SrF3Nxj86wPY4VbaOIungejla92/AFneMIPEu0hSLq5N2YrXrn4bc6wxZ4H4qBWYle0VINUSdPxnE4VrUW4xT8UGf4Z2yNZeiycb0Yh5mipRqibo7At/Ar1boK38GhRr8u7CkqV0TdTMZpGKNaa3AG5okYIUXU0UtxmOr9BheIGCFF1NUHsIXqzcVfRYyAIurq6Tha9R7CaSJGQBF1dhyernrfwY9FDLMi6mwaTkWXaq3EJ7BYxDAqou4OwgGqdz2+JWIYFVF3DXwI01SrG6fjQRHDpIjRYHe8WfXuwukihkkRo8UHsK3qnY9fixgGRYwWW+FDqrcYH0OviBYrYjR5I/ZWvavwXREtVsRoMgEnYBPVWoOPY5GIFipitHkZDlG9P+ALIlqoiNHog5ipemfhNhEtUsRotAPeo3qP4hPoFdECRYxWR2M31bsY14hogSJGq6k4ERNUayU+huUihqiI0ewA7K96P8c3RAxREaPZeJyIaar3cdwvYgiKGO12w9Gq93d8Br0iBqmITnAsdlC9C/ArEYNURCeYiQ+q3mP4JNaIGIQiOsXr8DLVuwyXiBiEIjrFZJyAiap3CuaLGKAiOsneeKPq/QlfEDFARXSaE7GV6p2Hm0UMQBGdZlscr3r34yz0iOinIjrRW7C76n0dPxHRT0V0omn4EBqq1YOTsUxEPxTRqQ7Aq1Xvv/FvIvqhiE7VhVMxXfXOxp0iNqKITvY0nKB6t+MsrBaxAUV0ui21h3kiNqKITjYPn1S9TTEHY0VsQBGd7Az8RfUOx+4iNqKITvVbfFn1ZuN9IvqhiE7Ui9OwSPVOwFYi+qGITvQDXKF6L8QbRfRTEZ1mMT6JVao1CR/EJBH9VESnOQ+/Ub0DsZ+IASiik9yJM1VvBj6ELhEDUEQn+SQeUr2jsbOIASqiU/wXvq16O+E4EYNQRCdYiY9hmWo18AFsKmIQiugE38a1qrc3XidikIoY7R7Gaao3AR/EZBGDVMRodybuVL1D8WIRQ1DEaPY7fFn1tsSHRAxREaNVDz6FBar3bmwvYoiKGK2uwL+r3m44QkQLFDEaLcYpqjcGx2FLES1QxGj0ZfxW9V6CfxHRIkWMNn/B51RvEk5BEdEiRYw2Z+NvqvdWPE9ECxUxmtyAr6jetniXiBYrYrRYiQ9jteq9EzuKaLEiRotv46eq90wcLWIYFDEa3I+z0Kt6H8EUEcOgiNHgPNyseq/FASKGSRF1dwvOUb1N8T40RAyTIuruVCxUvTfieSKGURF1dikuVb1tMUfEMCuirubjM1itesdjGxHDrIi6uhA3qN7zcbiIEVBEHd2DM1RvIk7EZBEjoIg6+jTuVb0D8VIRI6Qh6ubn+IbqTccJGKcz/Bo3odi4HkzEgZgiWqYh6mQZPoGlqnc0/kFnmI834jb9NxXPxxTRMkXUyfdwjertgHfrHGfjNgMzHkW0VBF1MR+fwhrV+wBm6gw34yuiLRRRF3Nxq+rtg9fpDL04Ew+ItlBEHdyCL6reeByPKTrDdbhItI0i2l0vTsOjqvc67KszrMHJWCnaRhHt7nJcqnozcJLOcT5+IdpKEe1sGT6FFap3HHbUGe7F2egVbaWIdnYBblC9XfE2neOL+LNoO0W0q7/jU6rXhTnYUme4GeeKtlREu/o07lO9F+NQneOjWCjaUhHt6AZ8XfXG4qMYpzP8CJeItlVEu1mD07BE9d6G5+sMj+FTWCPaVhHt5nu4SvWehHfrHN/AL0VbK6KdLMApqtfAidhJZ3gIn0GPaGsN0U7m4jbVm4JV+CZWG93G4nL8TbS9hmgXt+Bc7WEhjhXRZopoF6fjYe2hV0QbKqId/Ce+JSI2qIiqLcfJ6BYRG1RE1b6KX4qIjSqiSnfjLBHRL0VU6RzcJSL6pYiq3Igvioh+K6IKa3AyVoiIfiuiCt/Ff4iIASlipM3D6egREQNSxEj7Km4SEQNWxEi6DWeKiEEpYiR9HI+KiEEpYqRcjYtFxKAVMRIW4dNYJSIGrYiR8B1cJyKGpIjhdh9OFxFDVsRwOxN3ioghK2I4/Q/OFxEtUcRwWYFPYrGIaIkihsuluEJ0qh7RckUMhwfwUawRnaoLPaKlGmI4rME7MV50oh5MwuaipRpiODwJx4iIlioiImqiiIioiSIioiaKiIiaKCIiaqKIiKiJIiKiJoqIiJooIiJqooiIqIkiIqImioiImigiImqiiIioiSIioiaKiIiaKCIiaqKIiKiJIiKiJoqIiJooIiJqooiIqIkiIqImioiImigiImqiiIioiSIioiaKiIiaKCIiaqKIiKiJop66RETHKeppoogYqrFqpqGevoM7sEZEDFQXxuA3aqYLent71c2xp3zZ+d/5selTJ4uIgZu/cLEL585x8CteoC6KiIiaKCIiaqKIiKiJIiKiJoqIiJooIiJqooiIqIkiIqImioiImigiImqiiIioiSIioiaKiIiaKCIiaqLoPF06Qxe61EeXiI1oGN22xFOxI7bDdIxHN5ZiAe7B3bgLf1NP22M7PAWzMBUTPG45lmAe7sa9uAPzVaOBnTAbT8HWmIpx6MESPII7cBf+ivki0DA6/SPehH/C0zDJhi3FXbgZV+GHWGjgpuClmIE11hqHX+IPWmdzHIT9sAtmY6yNewR34EZcjmux2vDbHq/Bi/A0zEKXDXsUf8UvcRmuQ6/W2ASvxAT0GrhuLMUjuBt/MzjPxm5YZQTNX7h47IVz5/wn7lYjDaPLpvggjsQ0/TcZu2AXvA534mu4AA/ovy3xSeyAXmsVnIQ/GLqZeDvejO0M3GbYDLvjrbgJ5+ASrNB6O+JdOBSbGZhNsSl2x5H4JT6Lq9FtaDbDeZiCXgPThV6swSosw924HpfiN1imfw7FHPQYOV3owmG4W40Uo8czcSXmYJrBK9gep+E6vBXj9E8PutCFgoLicb2GpuAwXIdTsZ2hm4g98G38AM/TOlNwAn6OY7CZoZmMfXA5voGnGppe9KALBQUFBQUFBQUFBQUFXSgYh02wBZ6L9+Jn+BFeg2Ljej2uoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgy+O61UzD6LArLsFsrbUjvor9cCJut3E91q/X4G2KT+JIdBke++E5+BTONDQ74izsr2+r8RfciYexCmOxGWZjJ0ywfv+CPfA+/MDg9RgeL8Ze+DaOx4P6NkZ1ipppqL8n4ALMNnwOxkwchEeNrG1xAV5s+G2GM/BMvA0rDNxz8XU81bpW4bf4Lq7B37EMq9CLLozDRGyFvXEI9sAEzbbDRTgeZ2uNBViIYsO6MBbjMRETrWsM3ojtcRjusn6P4EGs1D/dmIZNrdWLh7AcxcZ1YRwWq5mG+jsBz7Jha3AbHsYqjMV0bI0t9c/VeNTIeiIuxu76bz4exSqMwWRshon6byK6DNzu+HfMsq7r8Vlcih7r14uVWImFuBXn4hU4BvtpNg5noYEzDd2n8XmMQ5cNG49pmIXdsCf2xGTN9sC3cCAetq65+BJ69c8KvBWfQ5fHLcXhuAHj9E8XFquZhhqbPnXyTniDDfsWLsSteAQrMRbT8UTsjJfhlZhi/a7CmUbWNHwdu9u4BbgM1+KveASr0MAkbIYd8Dy8HFvr2/U4AssNzI74OmZpthyfwDlYZHCuxE9xFE7FVGsVfBwP4luGZgmWYqn+eQB/xo9xBp6P9+MAzfbAqXindS3DMgOzWLMeLMIS/TRj+hRvfM+ZLpw7R5001Nt+2Nr6rcFxOBfdmnXjQTyI3+Bb2AFH4i2YYa2HcBxWGFkfxz42bAXOxzn4K3r07Tqcj5l4A96J2ZrdjqOwyMBMxjnYSbNFOALfN3TLMRc34wI8yVrj8Tn8Bb82eA2DtwY/w6/wYZyo2VtwMf7L0I3RrAsNHaCoqfO/8+MGXqhvn8fn0W3juvFnvA/74LvWOgm3GVkH42027A4cgnfhL+ixcb14AJ/BXrgA3R73GN6BPxm492JfzRbgMHxfa12DQ3C/ZjPwaUxRrRX4EM7VbAIOxxgxaEV9TcXTrN9juNjg3IR/wftxNi40sqbjIxinb3/EK3G5wfsbjsAxmIcT8RMD9wy8R7NufABXGB7/g6OwTLO98Sbt4VTcodl+mCkGraiv8djU+j2Ehw3eGpyBOVhpZB2BXfTtPhyGW7XGeXgBzjNwXZiDGZp9E18xvC7HGZp14T3YTPUewtc0m4lniEEr6qtgjPWbgPGGrtvImoZ36NtyvBd/1Fp/RbeBezZeqdn9ONXIOAO/0Wx7vE57+AVWavYMMWhFfa3Baus3Ezuqn1fjyfr2ffxA+3glZmh2Pu4yMhbjs+jV7HA0VO8B3K/ZVmLQivpaiYet31i8H1uojy4cgGL9HsNZWKM9TMVBmi3AvxlZV+F3mj0Dz1G9pVii2SQxaEV9LcWf9e35+B72VA9PwTP17T/xO+1jZ+yi2dX4u5E1H1drNhEvVr2CotlqMWhFTR156L6rcb0N+yf8CN/EazBF+9oJ2+rbxdrLHujS7CfoNvKuwXLNnoWiWtMwQ7P5YtCKersa99iwqTgMF+E3+BIOxnaYpH08HQ3r9yj+W3t5lmaLcItq3ISHNNsWm6rW9pip2e1i0BpqbOFjS2+fPnXy5/EZGzcOO2AHvB0L8Dv8Ejfg97hXdWbr2y1YYGAmYxp6DE7BYiy2roLtNLsP96vGAvwVs621ObbEw6rzGnRZayVuEoPWUH9fwG54vYF5Al6MF6Mbt+B6XIJr0WvkjMUW+nYPVhiYQ3AsegzOGHwDZ1nXZGyu2QIsUJ2/aTYd01Vnb7xas//GHWLQGupvOY7AEhyBYuDGYFfsijfj1/gsrsAaw28spunbo+g2MFvjmYZmtvWbiE00W4ZlqvOIZuMxQTW2xecxSbOLsFwMWjE6LMdROAI3G5pJ2AuX4kLsYPh1Yay+rTZwqwzdKus3Bl2adaNHdVZp1oUuI+85+C521ux/cIkYkmL06MXX8Aoch98aukNxJV5oePWiW9/GaS/d6NWsgTGqM1azXvQYmNUGbxZOwg/xHM2W4SQ8LIakYfT5O+bi37AHXo19sBUmGbjtcTEOwO8MjzVYrG9PQEGP/usyfFZimWaTMQmLVGOGZsux3MBsipko6NK3MZiELfBUvAh7Ymvr6sZ7cY0YsobRaxGuxtXYBLvj+XgWdsKOKPpna5yFA/GY1luNh/VtFiZiqf5rGD5L8AiebK1NsRkWqcYszRZggYH5AN6NYsMKxmGSDXsEJ+CroiUaOsMSXItrPW47PB0vwMvxjzZuT7wGX9N6vbhP356KqViq/y7FfejRt+V4Dt6Pov+6cQ+ea61t8ETcYeRtgqdoNg8PGZjJmKw1foKT8QvRMg2d6S7chSvwabwA/4qXo1i/gkNwEVZqvT/r20w8Ew/ov1txq417EHNQDMzvcIi1JuEf8DMjb1dsrdmdmG9kLcHv8SX8AEtFSzXEIlyJK3EETscM6/dcbI57td7tmIctrN9rcZXWm4wuA3eDdb0U52KNkbUPJlmrF78ycKuwAsXGrcYSPIx78VtcixvQI4ZFQ/xvX8V4nINiXZthFu7Ven/GrdjC+u2P2bhbe7gZt2MHa+2Dp+GPRs5U7KvZElxr4L6I76PLxq3GEjyCB8WIaIj/62IchV2t3+aGxzJch72s30y8C3O0h0fxI7zXWpPwDvyrkfMiPF+zX+EPBu5WXC/aVjE6TNM6j+JefRtj+HwXC/Xtzdhb+7gESzV7A/7RyJiA41A0+4rBmSjaWlF/s3AZjtEaBV36tszwuQWX69sMfA7baJ0eg/crXKnZVJyKyYbfUdhLsxtxlRiVihqbPnXyeHwOe+FszMVUQzMT21m/btxveJ2B5fr2DFyIbbTGWIO3CnOxVLNXYo7htRdOta5P4zExKhX19nEc6HEFx+JKvMzgvR5PtX534X7D6/c424btjcuwt6F5IU7AGIN3A86zrg/jSMNjZ3wJ0zT7d/xIjFpFTZ3/nR8fiWOt6wX4Ab6GvQzM4fiwvv0cCwy/z+DXNmw3XIIz8VQDsycuwOXYU9+69M/H8WvNGjgbx2it3fF97KTZ3TgBK8Wo1VBDx57y5f1xHsZYv0k4HAfjRvwQ1+MerMBq9GIsJuLZeAsORMP69eL76Db85uOduAJb6Nt0vBdvwnW4EjfhQazyuILpeAp2x77YFZvYuHH6ZwHegUvxJGtNwuewMz6MRwzeWLwen8UMzZbhbbhLjGoN9bQC92GWDZuMvbAXenEfHsIC9GAatsVMG3c5fmrk3Ig345uYYcM2w8E4GD2Yh8XowUTMwCYG5mFco/9+i7fgImxurS68Ay/EXPwI8/RfwQvxLhxsXUtwFK4Ro15RQ2ef8vZr8XL8VP91YRs8Cy/BvtgdM23cPJyEZUbWVXgT7td/BTOxA3bCLGxiYG7Ea3CZgbkWr8c91rULzscV+Bj2webWbxL+Af+KS3ElDrauR/E2fFt0hIaaOvLQfW/93pW/OAjH41hsYngswVH4g2pcgQNxDp5neC3D+TgV8w3OtXgVzsMe1vVsPBvvxSP4O+ZhKcZiU2yDmZiubzfhWPxMdIyGGlv42NLHpk+dfBKuwvuwP8ZqndvwHlytWjdiP5yAt2JzrbUG1+AMXGvo/oBXYA7eji2saxJmYZaBWYgLcRrmiY5SjA6/wCE4CBdhqaFZhi9jf1ytPSzECXgFzsUCQ7cEF+O1eBWu1ToL8WHsjy9hkaFZim9gf7wb80THaaipZctXWrZoif9lDa7Ej7ED9sfLsAumYaK+9WIJ7sVV+CZ+Z+DGYIr1m6g1bsSNOBMH4AA8A9MwwYYtw2L8Hv+Bq3A71hg+N+JGnIWDcACehqkYp2+r8Rj+gitwGf6EHq1RME2zCephPLqsNQUNA7Ri0RKrVq9RJw01tdfzdvH/TZo43v+xBrfiVpyBbbAztscWmIrxHrcKC3AfbsbvsdzgLcBXsA1WWWsSfqG17sDZOBuzsDO2xxaYinHoxQoswoO4A7fgASPvLzgdp+PJ2BXbYQtMwRh0Ywkewd34I24zPBZhLjZBL8bjV+rh9zgXBV1Yhr8boKXLVthh9tbqpAt6e3tFRLS7IiKiJoqIiJooIiJqooiIqIkiIqImioiImigiImqiiIioiSIioiaKiIiaKCIiauL/AdND5+mbMcvfAAAAAElFTkSuQmCC)
# 
# In this blog post I'm going to contrast our previous classifier example with another gradient-boosted tree model, a regression. First off, what's the difference between classification and regression?
# 
# A *classifier* looks at a row of data, and based on training data, identifies a class for it. This could as easily be fraud or "not fraud", "spam" or "ham," or even one of many dog breeds, depending on your dataset.
# 
# Meanwhile, a *regressor* attempts to predict a value or a score based on a number of factors or features, based on weights it learned from the training dataset.
# 
# Read on, as we'll attempt to predict California housing prices based on a standard dataset, with a similarly parametrized model to the one we use in the previous classficiation blog post. I'll explain how to track runs and experiments, as well as report multiple metrics. While you can optimize against either one or two metrics, today we'll just optimize against one and store the other for later use.
# 
# If you don't have XGBoost already installed in your Python or Anaconda environment, be sure to uncomment the first line. Then, let's start out by importing some useful libraries:

# In[14]:


#!conda install -y -c conda-forge xgboost
#%config InlineBackend.figure_formats = ['retina'] # optional, in case you'd like to add high DPI charts

import pandas as pd
import numpy as np
import seaborn as sns
import time

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import xgboost as xgb

import warnings
warnings.filterwarnings("ignore")  # don't spam the notebook with warnings


# ## Starting with data:
# To keep things simple, we’ll use a standard, cleaned dataset that exists as part of scikit-learn to train our model: this time we'll use the California housing dataset. It consists of 30 numerical properties (or “features”) that predict whether a certain observation in a scan represents cancer or not, either “malignant” or “benign.” This output representation is the “target” or “label” for each row, representing an individual patient’s outcome.
# 
# If you uncomment it, the first line should install a recent version of the XGBoost package from conda-forge, in case you don’t have it in your environment already:

# In[10]:


ch_dataset = fetch_california_housing()
data, target = fetch_california_housing(return_X_y=True)
X = pd.DataFrame(ch_dataset.data)
X.columns = ch_dataset.feature_names
y = target

#Split data into 3: 60% train, 20% validation, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1337)

print(X.columns)


# By printing the dataset column names, we can see that the features for our regression include standard properties of a house, including number of rooms by type, income, location, and more. Conveniently, all of these values have been translated to numerical values, so we don't have to do any transformation.

# ## Setting a baseline:
# 
# Let's train our regression with a somewhat rational set of parameters to start. This will serve as our baseline, after which, with SigOpt and Bayesian Optimization, we'll attempt to train a more accurate predictor of house price.
# 
# The following code defines an XGBoost regression model and fits the model against the training data we just imported.

# In[15]:


gbm = xgb.XGBRegressor( 
                        n_estimators=30000,
                        max_depth=4,
                        learning_rate=.05, 
                        subsample=.8,
                        min_child_weight=3,
                        colsample_bytree=.8
                       )

eval_set=[(X_train,y_train),(X_val,y_val)]
fit_model = gbm.fit( 
                    X_train, y_train, 
                    eval_set=eval_set,
                    eval_metric='rmse',
                    early_stopping_rounds=50,
                    verbose=False
                   )

gbm.score(X_test, y_test)


# At around 0.85, this "score," or a R^2 value, indicates fairly high accuracy, although it also obfuscates the average error. While it is one metric for how a model performs, we'll also look at RMSE or root-mean squared error, which is actually the metric we'll ask SigOpt to minimize.
# 
# Speaking of optimization, let's set up SigOpt to optimize this regressor. In order to retrieve your API key, you'll need to [sign up here](https://app.sigopt.com/signup). It's free, so please go ahead and do that now. Once you have your key, paste it in the cell/block below:

# In[16]:


# Install SigOpt's client library
get_ipython().system('pip install sigopt')
import sigopt
 
# Create a connection to SigOpt using either your Development or API token
from sigopt import Connection
 
# put your token here
api_token = "YOUR_API_TOKEN_HERE"
 
conn = Connection(client_token=api_token)


# Next, in order to kick off the experimentation loop, you'll need to set up one Python functions that creates the model with all of the parameters you want to tune, and another that evaluates that model. Note that these are fairly similar in terms of parameters and search space as we used in our prior classification example, but that this model returns two metrics, RMSE and an accuracy score, which the scikit-learn documentation defines to be R^2:

# In[18]:


def create_model(assignments):
    model = xgb.XGBRegressor(
                                n_estimators     = assignments['n_estimators'],
                                min_child_weight = assignments['min_child_weight'],
                                max_depth        = assignments['max_depth'],
                                gamma            = assignments['gamma'],
                                subsample        = assignments['subsample'],
                                colsample_bytree = assignments['colsample_bytree'],
                                reg_lambda       = assignments['lambda'],
                                reg_alpha        = assignments['alpha'],
                                learning_rate    = assignments['log_learning_rate'],
                                n_jobs           = 4
                            )
    return model

def evaluate_model(assignments):
    model = create_model(assignments)
    probabilities = model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(preds, y_test))
    
    return [
        dict(name="Accuracy Score", value=model.score(X_test, y_test)),
        dict(name="RMSE", value=rmse)
        ]


# Note that rather than returning a singular metric, we return a dictionary comprising of two metrics. In the next block we'll specify which to optimize and which to merely track. Hanging onto metrics can be useful if you want to revisit a different aspect of your modeling problem later on in the development process.
# 
# Let's next create our experiment and set up all of the parameters and metrics to be tracked in the SigOpt dashboard:

# In[19]:


experiment = conn.experiments().create(
    
    name="CA House Pricing XGB - Vanilla SigOpt",
 
    parameters=[
        dict(name="n_estimators", bounds=dict(min=50,max=350), type="int"),
        dict(name="min_child_weight", bounds=dict(min=2,max=15), type="int"),
        dict(name="max_depth", bounds=dict(min=3,max=10), type="int"),
        dict(name="gamma", bounds=dict(min=0,max=5), type="double"),
        dict(name="subsample", bounds=dict(min=0.5,max=1), type="double"),
        dict(name="colsample_bytree", bounds=dict(min=0.5,max=1), type="double"),
        dict(name="lambda", bounds=dict(min=0.00001, max=1), type="double", transformation="log"),
        dict(name="alpha", bounds=dict(min=0.00001, max=1), type="double", transformation="log"),
        dict(name="log_learning_rate", bounds=dict(min=0.00001 ,max=1), type="double", transformation="log")
        ],
 
    metrics=[
        dict(name="Accuracy Score", objective="maximize", strategy="store"),
        dict(name="RMSE", objective="minimize", strategy="optimize")
        ],
 
    observation_budget = 120,
)
 
print("Explore your experiment: https://app.sigopt.com/experiment/" + experiment.id + "/analysis")


# 
# On the above page, you can start to see metrics come in for both RMSE and R^2 Score as SigOpt suggest better and better paramter sets, resulting in a higher performing model. Feel free to explore parameter importance and parallel coordinates.
# 
# *Note:*
# 
# On a 4-core system, the following set of 120 observations will take roughly 10 minutes to execute.

# In[20]:


#Optimization Loop
for _ in range(experiment.observation_budget):
    suggestion = conn.experiments(experiment.id).suggestions().create()
    assignments = suggestion.assignments
    value_dicts = evaluate_model(assignments)
 
    conn.experiments(experiment.id).observations().create(
        suggestion=suggestion.id,
        values=value_dicts
    )
    
    #update experiment object
    experiment = conn.experiments(experiment.id).fetch()
 
assignments = conn.experiments(experiment.id).best_assignments().fetch().data[0].assignments  
 
print("BEST ASSIGNMENTS FOUND: \n", assignments)


# At the bottom of your notebook or interpreter’s output, you should see the best set of parameters SigOpt was able to find in 120 automated training runs.
# 
# ## Background on XGBoost’s parameters:
# * `min_child_weight`, used to control over-fitting, this parameter is the sample size under which the model can not split a node. Higher values prevent a model from learning relations which might be highly specific to the particular sample selected for a tree.
# * `max_depth`, this is the maximum depth of a tree. This parameter controls over-fitting as higher depth will allow the model to learn relations very specific to a particular sample.
# * `gamma`, this parameter specifies the minimum loss reduction required to make a split. The larger gamma is, the more conservative the algorithm will be. subsample, defines the ratio of the training instances. For example setting it to 0.5 means that XGBoost would randomly sample half of the training data prior to growing trees, preventing overfitting
# * `colsample_bytree`, defines the fraction of features to be randomly sampled for each tree. alpha and lambda are the L1 and L2 regularization terms on weights. Both values tend to prevent overfitting as they are increased. Additionally, Alpha can be used in case of very high dimensionality to help a model training converge faster
# * `learning_rate` controls the weighting of new trees added to the model. Lowering this value will prevent overfitting, but require the model to add a larger number of tree
# 
# ## Wrapping up:
# Today we explored regression on an example dataset, but you can also apply these techniques to predict all kinds of numerical outcomes based on historical training data. You can easily discover for yourself how SigOpt facilitates a robust, efficient, and well-tracked experimentation process. Aside from ease-of-use, SigOpt delivers much better optimization performance, when compared with human-tuned parameters or an exhaustive approach like grid search. If you’d like, you can find the original iPython Jupyter Notebook code for this example here, or you can run it on Colab here. 
# 
# If you'd like to head back to the previous post, covering [classification](https://sigopt.com/blog/xgboost-classification-optimization/) instead of regression, [you can find that here](https://sigopt.com/blog/xgboost-classification-optimization/).