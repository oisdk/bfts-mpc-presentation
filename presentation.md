---
title: Breadth-First Traversals Via Staging
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

- Takeaway: a technique for **staging** effectful computations, using *Phases*

    * The ability to **reorder** effects **without** reordering the expressions
      that give rise to those effects.
    
- In the paper:

    * Theory and implementation of the Phases type

    * With some of the theory of *Applicatives*, *Traversables*, and *Free
      Applicatives*
      
- In this talk:

    * How to **use** *Phases* for *sort-tree*, *breadth-first traversal*...

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

. . .

```haskell
instance Monad f => Monad (Phases f) ✗
```

. . .

```haskell
main = (putStr "What's your name? " *> getLine) >>= 
       \n -> putStr ("Hello, " ++ n)
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
n /= m ==> 
  phase n x ⊗ phase m y == twist <$> (phase m y ⊗ phase n x)
```

```haskell
twist :: (a, b) -> (b, a)
```

. . .

```haskell
n /= m ==>
  phase n x <* phase m y == phase m y *> phase n x
```

. . .

```haskell
n /= m ==> phase n x `CommutesWith` phase m y
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
 φ . traverse f = traverse (φ . f)
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

## Sequential Fusion and Commutativity

```haskell
f `CommutesWith` g ==>
  traverse f t *> traverse g t = traverse (\x -> f x *> g x) t
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

# Breadth-First Traversals

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

## Depth-First Traversal

```haskell
traverse :: Applicative f => (a -> f b) -> Tree a -> f (Tree b)
traverse f (x :& xs) = (:&) <$> f x <*> traverse (traverse f) xs
```

## Depth-First Traversal

```haskell
traverse :: Applicative f => (a -> f b) -> Tree a -> f (Tree b)
traverse f (x :& xs) = ⦇ f x :& traverse (traverse f) xs ⦈
```

. . .

```haskell
fmap :: (a -> b) -> Tree a -> Tree b
fmap f (x :& xs) = f x :& fmap (fmap f) xs
```


## Level-Wise Enumeration

```

   ↓     ↓     ↓
 ┏━━━┓ ┏━━━┓ ┏━━━┓
 ┃ 3─╂┬╂─1─╂┬╂─1 ┃
 ┗━━━┛│┃   ┃│┃   ┃
      │┃   ┃│┃   ┃
      │┃   ┃└╂─5 ┃
      │┃   ┃ ┃   ┃
      │┃   ┃ ┃   ┃
      └╂─4─╂┬╂─9 ┃
       ┗━━━┛│┃   ┃
            │┃   ┃
            └╂─2 ┃
             ┗━━━┛

[[3],[1,4],[2,5,9,2]]
```

## Labelling With Level

```haskell
        ╭                         ╮
relevel │ 3 :& [ 1 :& [ 1 :& []   │ = 0 :& [ 1 :& [ 2 :& []
        │             , 5 :& []]  │               , 2 :& []]
        │      , 4 :& [ 9 :& []   │        , 1 :& [ 2 :& []
        │             , 2 :& []]] │               , 2 :& []]]
        ╰                         ╯
```

. . .

```haskell
relevel :: Tree a -> Tree Int
relevel = go 0 where
  go n (x :& xs) = n :& map (go (n+1)) xs
```


## Combination

```haskell
traverse :: Applicative f => (a -> f b) -> Tree a -> f (Tree b)
traverse f = go where
  go (x :& xs) = (:&) <$> f x <*> traverse go xs
```

```haskell
relevel :: Tree a -> Tree Int
relevel = go 0 where
  go n (x :& xs) = n :& map (go (n+1)) xs
```

. . .

```haskell
bft :: Applicative f => (a -> f b) -> Tree a -> f (Tree b)
bft f = runPhases . go 0 where 
  go n (x :& xs) = (:&) <$> phase n (f x) <*> traverse (go (n+1)) xs
```

## Breadth-First Renumbering

```haskell
renumber t = evalState (bft num t) 1 where num _ = get <* modify succ
```

```haskell
         ╭                         ╮
renumber │ 3 :& [ 1 :& [ 1 :& []   │ = 1 :& [ 2 :& [ 4 :& []
         │             , 5 :& []]  │               , 5 :& []]
         │      , 4 :& [ 9 :& []   │        , 3 :& [ 6 :& []
         │             , 2 :& []]] │               , 7 :& []]]
         ╰                         ╯
```

# Questions?
