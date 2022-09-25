---
title: Breadth-First Traversals via Staging
patat:
  theme:
    codeBlock: [vividBlack]
    code: [vividBlack]
  incrementalLists: true

...

# Blank

## Breadth-First Traversals via Staging

- Jeremy Gibbons, *Oisín Kidney*, Tom Schrijvers, and Nicolas Wu
```
       ^             ^               ^               ^
       ┃             ┃               ┃               ┃
    Oxford           ┃           KU Leuven           ┃
                     ┃                               ┃
                     ┃                               ┃
                     ┗━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┛
                                  Imperial
```

- MPC 2022

# Introduction

## A Real-World Problem

- CLIENT: "I would like the letters 'a' to 'e' printed out!"

- PROGRAMMER:

```haskell
main = do print "a"
          print "b"
          print "c"
          print "d"
          print "e"
```

- CLIENT: "No! I want them in this order: **bedca**"

- PROGRAMMER:

   ```haskell
   main = do print "b"
             print "e"
             print "d"
             print "c"
             print "a"
   ```

- CLIENT: "Yuck! I want the **source code** to have the letters in alphabetical
  order!" 
  
- PROGRAMMER (has just read "*Breadth-First Traversals Via Staging*"):

. . .

```haskell
main = runPhases $ do phase 5 (print "a")
                      phase 1 (print "b")
                      phase 4 (print "c")
                      phase 3 (print "d")
                      phase 2 (print "e")
```

. . .

```haskell
>>> main
b
e
d
c
a
```

- CLIENT: "Good! Here is a raise. Your next assignment is to figure out the
  **category theoretic principles** underlying that piece of code."
  
## Overview

- Takeaway: a technique for **staging** effectful computations

    * The ability to **reorder** effects **without** reordering the syntactic
      expressions that give rise to those effects.
      
    * Alternatively: the ability to reorder the **effect** part of effectful
      computations **without** reordering the **pure** part.
      
- Applying this technique to problems like *repmin*, *sort-tree*, and
  culminating in *breadth-first traversal*

- Using some of the theory of *Applicatives*, *Traversables*, and *Free
  Applicatives*

## Important Types

. . .

```haskell
data Tree a = a :& [Tree a]
```

. . .

```haskell
  3 :& [ 1 :& [ 1 :& []            --      3─┬─1─┬─1
              , 5 :& []]           --        │   └─5
       , 4 :& [ 9 :& []            --        └─4─┬─9
              , 2 :& []]]          --            └─2
```

## Traversal Orders

```
   breadth-first                depth-first


   3──┬──1──┬──1               3──┬──1──┬──1
      │     │                     │     │
      │     │                     │     │
      │     └──5                  │     └──5
      │                           │
      │                           │
      └──4──┬──9                  └──4──┬──9
            │                           │
            │                           │
            └──2                        └──2
```

## Traversal Orders

```
   breadth-first                depth-first
   ↓     ↓     ↓
 ┏━━━┓ ┏━━━┓ ┏━━━┓           ┏━━━━━━━━━━━━━━━┓
 ┃ 3─╂┬╂─1─╂┬╂─1 ┃         → ┃ 3──┬──1──┬──1 ┃
 ┗━━━┛│┃   ┃│┃   ┃           ┗━━━━┿━━━━━┿━━━━┛
      │┃   ┃│┃   ┃                │     │┏━━━┓
      │┃   ┃└╂─5 ┃         →      │     └╂─5 ┃
      │┃   ┃ ┃   ┃                │      ┗━━━┛
      │┃   ┃ ┃   ┃                │┏━━━━━━━━━┓
      └╂─4─╂┬╂─9 ┃         →      └╂─4──┬──9 ┃
       ┗━━━┛│┃   ┃                 ┗━━━━┿━━━━┛
            │┃   ┃                      │┏━━━┓
            └╂─2 ┃         →            └╂─2 ┃
             ┗━━━┛                       ┗━━━┛

   [3,1,4,2,5,9,2]            [3,1,1,5,4,9,2]
```

## Applicative and Traversable

. . .

```haskell
class Functor f => Applicative f where
  pure  :: a -> f a
  (<*>) :: f (a -> b) -> f a -> f b
  
```

. . .

```haskell
(⊗)  :: Applicative f => f a  -> f b  -> f (a, b)
(<*) :: Applicative f => f a  -> f () -> f a
(*>) :: Applicative f => f () -> f a  -> f a
```

. . .

```haskell
class Foldable t => Traversable t where
  traverse :: Applicative f => (a -> f b) -> t a -> f (t b)
```

# Renumbering

## Renumbering: The spec

```haskell
renumber :: Tree a -> Tree Int
```

```haskell
         ╭                         ╮
renumber │ 3 :& [ 1 :& [ 1 :& []   │ = 1 :& [ 2 :& [ 3 :& []
         │             , 5 :& []]  │               , 4 :& []]
         │      , 4 :& [ 9 :& []   │        , 5 :& [ 6 :& []
         │             , 2 :& []]] │               , 7 :& []]]
         ╰                         ╯
```

## Renumbering with Traverse

```haskell
renumber :: Tree a -> Tree Int
```

```haskell
get       ::                 State Int Int
modify    :: (Int -> Int) -> State Int ()
evalState :: State Int a -> Int -> a
```

. . .

```haskell
instance Traversable Tree where ...
```

. . .

```haskell
renumber t = evalState (traverse num t) 1
  where num _ = get <* modify succ
```

# Fusing traversals with Staging

## Fusion

```haskell
map f . map g = map (f . g)
```

. . .

```haskell
φ . traverse f = traverse (φ . f)
```

## Sorting Tree Labels

```haskell
sortTree :: Ord a => Tree a -> Tree a
```

```haskell
         ╭                         ╮
sortTree │ 3 :& [ 1 :& [ 1 :& []   │ = 1 :& [ 1 :& [ 2 :& []
         │             , 5 :& []]  │               , 3 :& []]
         │      , 4 :& [ 9 :& []   │        , 4 :& [ 5 :& []
         │             , 2 :& []]] │               , 9 :& []]]
         ╰                         ╯
```

## Sorting Tree Labels

```haskell
push   :: a -> State [a] ()
pop    :: State [a] a
```

. . .

```haskell
sortTree :: Ord a => Tree a -> Tree a
sortTree t = flip evalState [] $ traverse push t         *>
                                 modify sort             *>
                                 traverse (\_ -> pop) t
```
. . .

```haskell
tree =  3 :& [  1 :& [  1 :& []         stack = []
                     ,  5 :& []]
             ,  4 :& [  9 :& []
                     ,  2 :& []]]
```

## Sorting Tree Labels

```haskell
push   :: a -> State [a] ()
pop    :: State [a] a
```

```haskell
sortTree :: Ord a => Tree a -> Tree a
sortTree t = flip evalState [] $ traverse push t         *> -- <--
                                 modify sort             *>
                                 traverse (\_ -> pop) t
```

```haskell
tree =  3 :& [  1 :& [  1 :& []         stack = []
                     ,  5 :& []]
             ,  4 :& [  9 :& []
                     ,  2 :& []]]
```

## Sorting Tree Labels

```haskell
push   :: a -> State [a] ()
pop    :: State [a] a
```

```haskell
sortTree :: Ord a => Tree a -> Tree a
sortTree t = flip evalState [] $ traverse push t         *> -- <--
                                 modify sort             *>
                                 traverse (\_ -> pop) t
```

```haskell
tree = () :& [ () :& [ () :& []         stack = [3,1,1,5,9,4,2]
                     , () :& []]
             , () :& [ () :& []
                     , () :& []]]
```

## Sorting Tree Labels

```haskell
push   :: a -> State [a] ()
pop    :: State [a] a
```

```haskell
sortTree :: Ord a => Tree a -> Tree a
sortTree t = flip evalState [] $ traverse push t         *>
                                 modify sort             *> -- <--
                                 traverse (\_ -> pop) t
```

```haskell
tree = () :& [ () :& [ () :& []         stack = [3,1,1,5,9,4,2]
                     , () :& []]
             , () :& [ () :& []
                     , () :& []]]
```

## Sorting Tree Labels

```haskell
push   :: a -> State [a] ()
pop    :: State [a] a
```

```haskell
sortTree :: Ord a => Tree a -> Tree a
sortTree t = flip evalState [] $ traverse push t         *>
                                 modify sort             *> -- <--
                                 traverse (\_ -> pop) t
```

```haskell
tree = () :& [ () :& [ () :& []         stack = [1,1,2,3,4,5,9]
                     , () :& []]
             , () :& [ () :& []
                     , () :& []]]
```

## Sorting Tree Labels

```haskell
push   :: a -> State [a] ()
pop    :: State [a] a
```

```haskell
sortTree :: Ord a => Tree a -> Tree a
sortTree t = flip evalState [] $ traverse push t         *>
                                 modify sort             *>
                                 traverse (\_ -> pop) t    -- <--
```

```haskell
tree = () :& [ () :& [ () :& []         stack = [1,1,2,3,4,5,9]
                     , () :& []]
             , () :& [ () :& []
                     , () :& []]]
```

## Sorting Tree Labels

```haskell
push   :: a -> State [a] ()
pop    :: State [a] a
```

```haskell
sortTree :: Ord a => Tree a -> Tree a
sortTree t = flip evalState [] $ traverse push t         *>
                                 modify sort             *>
                                 traverse (\_ -> pop) t    -- <--
```

```haskell
tree =  1 :& [  1 :& [  2 :& []         stack = []
                     ,  3 :& []]
             ,  4 :& [  5 :& []
                     ,  9 :& []]]
```

## Sorting Tree Labels

```haskell
push   :: a -> State [a] ()
pop    :: State [a] a
```

```haskell
sortTree :: Ord a => Tree a -> Tree a
sortTree t = flip evalState [] $ traverse push t         *>
                                 modify sort             *>
                                 traverse (\_ -> pop) t
```


**Can we do it with one traverse?**

## Phases Type

. . .

```haskell
data Phases f a where
  Pure :: a -> Phases f a
  Link :: (x -> y -> a) -> f x -> Phases f y -> Phases f a
```

. . .


```haskell
instance Applicative f => Applicative (Phases f) where ...
```

## Phases Type: usage

. . .

```haskell
runPhases :: Applicative f => Phases f a -> f a
phase     :: Applicative f => Int -> f a -> Phases f a
```

. . .

```haskell


runPhases $            do phase 4 (putStrLn "a")
                          phase 2 (putStrLn "b")
                          phase 3 (putStrLn "c")
                          phase 1 (putStrLn "d")
                          phase 2 (putStrLn "e")
```

## Phases Type: usage

```haskell
runPhases :: Applicative f => Phases f a -> f a
phase     :: Applicative f => Int -> f a -> Phases f a
```

```haskell


runPhases $            do phase 4 (putStrLn "a")    --     > d
                          phase 2 (putStrLn "b")    --     > b
                          phase 3 (putStrLn "c")    --     > e
                          phase 1 (putStrLn "d")    --     > c
                          phase 2 (putStrLn "e")    --     > a
```

## Phases Type: usage

```haskell
runPhases :: Applicative f => Phases f a -> f a
phase     :: Applicative f => Int -> f a -> Phases f a
```

```haskell
out s = putStrLn s *> pure s

runPhases $ sequenceA $ [ phase 4 (out      "a")    --     > d
                        , phase 2 (out      "b")    --     > b
                        , phase 3 (out      "c")    --     > e
                        , phase 1 (out      "d")    --     > c
                        , phase 2 (out      "e") ]  --     > a
```

. . .

```haskell
["a","b","c","d","e"]
```

## Phases Type: Commutativity

```haskell
                         n /= m
-------------------------------------------------------------
  phase n x ⊗ phase m y = twist <$> (phase m y ⊗ phase n x)
```

```haskell
twist :: (a, b) -> (b, a)
```


## Sorting Tree Labels

```haskell
push   :: a -> State [a] ()
pop    :: State [a] a
```

```haskell
sortTree :: Ord a => Tree a -> Tree a
sortTree t =
  flip evalState [] $
             traverse push t                      *>
             modify sort                          *>
             traverse (\_ -> pop) t
 ```
 
## Sorting Tree Labels

```haskell
push   :: a -> State [a] ()
pop    :: State [a] a
```
 
 ```haskell
sortTree :: Ord a => Tree a -> Tree a
sortTree t = 
   flip evalState [] $ runPhases $
     phase 1 (traverse push t)                     *>
     phase 2 (modify sort)                         *>
     phase 3 (traverse (\_ -> pop) t))
```

## Sorting Tree Labels

```haskell
push   :: a -> State [a] ()
pop    :: State [a] a
```
 
 ```haskell
sortTree :: Ord a => Tree a -> Tree a
sortTree t = 
   flip evalState [] $ runPhases $
     phase 2 (modify sort)                         *>
     phase 1 (traverse push t)                     *>
     phase 3 (traverse (\_ -> pop) t))
```

. . .

```haskell
traverse (φ . f) = φ . traverse f
```

## Sorting Tree Labels

```haskell
push   :: a -> State [a] ()
pop    :: State [a] a
```
 
 ```haskell
sortTree :: Ord a => Tree a -> Tree a
sortTree t = 
   flip evalState [] $ runPhases $
     phase 2 (modify sort)                         *>
             (traverse (\x -> phase 1 (push x)) t) *>
             (traverse (\_ -> phase 3 pop) t)
```

## Commutativity

```haskell
                    f x ⊗ g y = twist <$> g y ⊗ f x
-------------------------------------------------------------------------
  traverse f t ⊗ traverse g t = unzip <$> traverse (\x -> f x ⊗ g x) t
```

```haskell
twist :: (a, b) -> (b, a)
unzip :: f (a, b) -> (f a, f b)
```

## Sorting Tree Labels

```haskell
push   :: a -> State [a] ()
pop    :: State [a] a
```
 
 ```haskell
sortTree :: Ord a => Tree a -> Tree a
sortTree t = 
   flip evalState [] $ runPhases $
     phase 2 (modify sort)                         *>
             (traverse (\x -> phase 1 (push x)) t) *>
             (traverse (\_ -> phase 3 pop) t)
```

## Sorting Tree Labels

```haskell
push   :: a -> State [a] ()
pop    :: State [a] a
```
 
 ```haskell
sortTree :: Ord a => Tree a -> Tree a
sortTree t = 
   flip evalState [] $ runPhases $
     phase 2 (modify sort)                         *>
              traverse (\x -> phase 1 (push x)     *> 
                              phase 3 pop) t
```


<!--


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

---


---


```haskell
repminT :: (Traversable t, Ord a) => t a -> Day ((,) (BoundedAbove a)) ((->) (BoundedAbove a)) (t a)
repminT = traverse (\x -> phase1 (In x, ()) *> phase2 inBound)

runEnv :: Day ((,) e) ((->) e) a -> a
runEnv (Day c (e,xs) ys) = c xs (ys e)

repmin = runEnv . repminT
```

-->

# Breadth-First Traversals

## Breadth-First Enumeration

```haskell
levels :: Tree a → [[a]]
levels = go 0 where
  go n (x :& xs) = level n [x] <> foldr lzw [] (go (n+1) xs)
```

. . .

```haskell
lzw :: Monoid a => [a] -> [a] -> [a]
lzw (x:xs) (y:ys) = (x <> y) : lzw xs ys
lzw []     ys     = ys
lzw xs     []     = xs
```

. . .

```haskell
level :: Monoid a => Int -> a -> [a]
level 0     x = x
level (n+1) x = mempty : level n x
```

## Breadth-First Enumeration

```haskell
levels :: Tree a → [[a]]
levels = go 0 where
  go n (x :& xs) = level n [x] <> foldr lzw [] (go (n+1) xs)
```

```haskell
       ╭                         ╮
levels │ 3 :& [ 1 :& [ 1 :& []   │ = [[3],[1,4],[1,5,9,2]]
       │             , 5 :& []]  │
       │      , 4 :& [ 9 :& []   │
       │             , 2 :& []]] │
       ╰                         ╯
```

## Breadth-First Traversal

```haskell
bft :: Applicative f => (a -> f b) -> Tree a -> f (Tree b)
bft f = runPhases . go 0 where 
  go n (x :& xs) = (:&) <$> phase n (f x) <*> traverse (go (n+1)) xs
```

. . .

```haskell
renumber t = evalState (bft num t) 1 where num _ = get <* modify succ
```

. . .

```haskell
         ╭                         ╮
renumber │ 3 :& [ 1 :& [ 1 :& []   │ = 1 :& [ 2 :& [ 4 :& []
         │             , 5 :& []]  │               , 5 :& []]
         │      , 4 :& [ 9 :& []   │        , 3 :& [ 6 :& []
         │             , 2 :& []]] │               , 7 :& []]]
         ╰                         ╯
```

# Questions?
