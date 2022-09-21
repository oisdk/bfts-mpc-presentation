---
title: Breadth-First Traversals via Staging
author: Jeremy Gibbons, Donnacha Oisín Kidney, Tom Schrijvers, Nicolas Wu
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
tree = 
  3 :& [ 1 :& [ 1 :& []            --      3─┬─1─┬─1
              , 5 :& []]           --        │   └─5
       , 4 :& [ 9 :& []            --        └─4─┬─9
              , 2 :& []]]          --            └─2
```

---

# Enumerations and Traversals

```haskell
dfe :: Tree a -> [a]
dfe (x :& xs) = x : concatMap dfe xs
```

```haskell
tree = 
  3 :& [ 1 :& [ 1 :& []            --      3─┬─1─┬─1
              , 5 :& []]           --        │   └─5
       , 4 :& [ 9 :& []            --        └─4─┬─9
              , 2 :& []]]          --            └─2
```

```haskell
dfe tree == [3,1,1,5,4,9,2]
```

---

# Enumerations and Traversals

We're not just interested in enumeration, we're interested in *traversal*

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

"Essence of the Iterator Pattern"

```haskell
instance Traversable Tree where
  traverse f (x :& xs) = (:&) <$> f x <*> traverse (traverse f) xs

```

---

# Enumerations and Traversals

* Okasaki, Chris. ‘Breadth-First Numbering: Lessons from a Small Exercise in
  Algorithm Design’. ICFP 2000.

---

