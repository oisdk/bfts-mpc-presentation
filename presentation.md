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

```haskell
data Day f g a
  = Day (x -> y -> a) (f x) (g y)
  
instance (Applicative f, Applicative g) => Applicative (Day f g) where
  pure x = Day (\_ _ -> x) (pure ()) (pure ())
  Day f xl yl <*> Day x xr yr = 
    Day (\(xl,xr) (yl,yr) -> f xl yl (x xr yr)) ((,) <$> xl <*> xr) ((,) <$> yl <*> yr)

phase1 :: Applicative g => f a -> Day f g a
phase1 x = Day const x (pure ())

phase2 :: Applicative f => g a -> Day f g a
phase2 = Day (const id) (pure ())
```

---

# Commutativity

---


```haskell
repminT :: (Traversable t, Ord a) => t a -> Day ((,) (BoundedAbove a)) ((->) (BoundedAbove a)) (t a)
repminT = traverse (\x -> phase1 (In x, ()) *> phase2 inBound)

runEnv :: Day ((,) e) ((->) e) a -> a
runEnv (Day c (e,xs) ys) = c xs (ys e)

repmin = runEnv . repminT
```

---

# Multiple Phases

```haskell
data Phases f a where
  Pure :: a -> Phases f a
  Link :: (x -> y -> a) -> f x -> Phases f y -> Phases f a
```

Notice that this is the free applicative.

But the applicative instance we're going to give it isn't:

```haskell
instance Applicative f => Applicative (Phases f) where
  pure = Pure
  Pure f <*> xs = fmap f xs
  fs <*> Pure x = fmap ($x) fs
  Link f x xs <*> Link g y ys = Link (\(l,r) (ls,rs) -> f l ls (g r rs)) ((,) <$> x <*> y) ((,) <$> xs <*> ys)
```

```haskell
now :: f a -> Phases f a
now x = Link const x (Pure ())

later :: Applicative f => Phases f a -> Phases f a
later = Link (const id) (pure ())


phase :: Applicative f => Int -> f a -> Phases f a
phase 0 x = now x
phase n x = later (phase (n-1) x)
```

---

# Sorting Leaves

```haskell
pop  :: State [a] a
push :: a -> State [a] ()

sortTree :: Ord a => Tree a -> Tree a
sortTree t = flip evalState [] $ do
  traverse push t
  modify sort
  traverse (\_ -> pop) t
  
sortTree :: Ord a => Tree a -> Tree a
sortTree t = flip evalState [] (runPhases (
  phase 0 (traverse push t) *>
  phase 1 (modify sort) *>
  phase 2 (traverse (\_ -> pop) t)))

sortTree :: Ord a => Tree a -> Tree a
sortTree t = flip evalState [] (runPhases (
  phase 1 (modify sort) *>
  traverse (\x -> phase 0 (push x) *> phase 2 pop) t))
```

---

# Breadth-First Traversals

* Okasaki notice that breadth-first seems more difficult than depth-first.

* Let's look at how we do breadth-first enumeration.

```haskell
levels :: Tree a → [[a]]
levels (x :& xs) = [x] : foldr (lzw (++)) [] (map levels xs)

lzw :: (a -> a -> a) -> [a] -> [a] -> [a]
lzw f (x:xs) (y:ys) = f x y : lzw f xs ys
lzw _ []     ys     = ys
lzw _ xs     []     = xs

       ╭                         ╮
levels │ 3 :& [ 1 :& [ 1 :& []   │ = [[1],[1,4],[1,5,9,2]]
       │             , 5 :& []]  │
       │      , 4 :& [ 9 :& []   │
       │             , 2 :& []]] │
       ╰                         ╯
```

--- 

```

   3──┬──1──┬──1
      │     │
      │     └──5
      │
      └──4──┬──9
            │
            └──2
```

---

```
 ┏━━━┓ ┏━━━┓ ┏━━━┓
 ┃ 3─╂┬╂─1─╂┬╂─1 ┃
 ┗━━━┛│┃   ┃│┃   ┃
      │┃   ┃└╂─5 ┃
      │┃   ┃ ┃   ┃
      └╂─4─╂┬╂─9 ┃
       ┗━━━┛│┃   ┃
            └╂─2 ┃
             ┗━━━┛
```

---

```
 ╔═══╗ ╔═══╗ ╔═══╗
 ║ 3─╫┬╫─1─╫┬╫─1 ║
 ╚═══╝│║   ║│║   ║
      │║   ║└╫─5 ║
      │║   ║ ║   ║
      └╫─4─╫┬╫─9 ║
       ╚═══╝│║   ║
            └╫─2 ║
             ╚═══╝
```
