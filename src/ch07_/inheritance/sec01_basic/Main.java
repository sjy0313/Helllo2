package ch07_.inheritance.sec01_basic;

public class Main {

	public static void main(String[] args) {
		Parent parent = new Parent();
		Child child = new Child();
		
		System.out.println(parent == child);
		System.out.println(child); // ch07_.inheritance.sec01.Child@28a418fc
		System.out.println(parent); // ch07_.inheritance.sec01.Parent@5305068a
		// 값이 다름. 
		// false
		System.out.println(child instanceof Parent); // true
		// child 객체는 부모로 부터의 연장선임을 알려줌.
		// java : instanceof / python : isinstnace

	}

}
