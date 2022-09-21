---
title: Breadth-First Traversals via Staging
author: Donnacha Oisín Kidney
patat:
  theme:
    codeBlock: [vividBlack]
    code: [vividBlack]
  incrementalLists: true

...

# Important Types

```haskell
data Tree   a = a :& Forest a
type Forest a = [Tree a]
```

. . .

```haskell
3 :& [ 1 :& [ 1 :& []       -- 3─┬─1─┬─1
            , 5 :& []]      --   │   └─5
     , 4 :& [ 9 :& []       --   └─4─┬─9
            , 2 :& []]]     --       └─2
```

---

# Applicative and Traversable

```haskell
class Functor f => Applicative f where
  pure  :: a -> f a
  (<*>) :: f (a -> b) -> f a -> f b
```

```haskell
class Foldable t => Traversable t where
  traverse :: Applicative f => (a -> f b) -> t a -> f (t b)
```
