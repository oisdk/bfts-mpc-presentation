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
tree
  =
    1 :&                     --      ┌8
      [ 2 :&                 --    ┌4┤
          [ 4 :&             --    │ └9
              [ 8 :& []      --  ┌2┤
              , 9 :& []]     --  │ └5
          , 5 :& []]         -- 1┤
      , 3 :&                 --  │   ┌10
          [ 6 :&             --  │ ┌6┤
              [ 10 :& []     --  │ │ └11
              , 11 :& []]    --  └3┤
          , 7 :& []]]        --    └7
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
