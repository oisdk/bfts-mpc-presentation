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

We're not just interested in enumeration, we're interested in *traversal*.

For instance, renumbering.

```haskell

         ╭                         ╮
renumber │ 3 :& [ 1 :& [ 1 :& []   │ = 1 :& [ 2 :& [ 3 :& []
         │             , 5 :& []]  │               , 4 :& []]
         │      , 4 :& [ 9 :& []   │        , 5 :& [ 6 :& []
         │             , 2 :& []]] │               , 7 :& []]]
         ╰                         ╯
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

"Essence of the Iterator Pattern"

```haskell
instance Traversable Tree where
  traverse f (x :& xs) = (:&) <$> f x <*> traverse (traverse f) xs

```

---

# Renumbering

```haskell
renumber t = evalState (traverse num t) 1
  where
    num _ = get <* modify succ
```

---

# Fusing traversals with Staging

```haskell
       ╭                         ╮
repmin │ 3 :& [ 1 :& [ 1 :& []   │ = 1 :& [ 1 :& [ 1 :& []
       │             , 5 :& []]  │               , 1 :& []]
       │      , 4 :& [ 9 :& []   │        , 1 :& [ 1 :& []
       │             , 2 :& []]] │               , 1 :& []]]
       ╰                         ╯
```

. . .

```haskell
repmin :: Tree ℕ -> Tree ℕ
repmin t = fmap (const m) t
  where
    m = minimum (dfe t)
```

---

```haskell
repmin t = let (u, m) = aux t m in u
  where
    aux :: Tree Int -> a -> (Tree a, Int)
    aux (x :& xs) m = (m :& ys, minimum (x : ms))
      where
        (ys, ms) = unzip (map aux xs)
```

. . .

```haskell
repmin t = let (u, m) = aux t in u m
  where
    aux :: Tree Int -> (a -> Tree a, Int)
    aux (x :& xs) = (\m -> m :& ys m, minimum (x : ms))
      where
        (ys, ms) = unzip (map aux xs)
```

---

```haskell
instance Monoid m => Applicative (m ,) where
  pure x = (mempty, x)
  (fm, f) <*> (xm, x) = (fm <> xm, f x)

data BoundedAbove a = In a | Top

instance Ord a => Monoid (BoundedAbove a) where
  mempty = Top
  Top <> x = x
  x <> Top = x
  In x <> In y = In (min x y)
  
getBounded :: BoundedAbove a -> a
getBounded (In x) = x

minimum :: Ord a => Tree a -> a
minimum = getBounded . fst . traverse (\x -> (In x, ()))
```

---

```haskell
instance Applicative (a ->) where
  pure x e = x
  (f <*> x) e = f e (x e)
  
replace :: Tree a -> b -> Tree b
replace = traverse (\_ e -> e)
```

---

```haskell
repmin t = replace t (minimum t)
```

--- 

# Day convolution
