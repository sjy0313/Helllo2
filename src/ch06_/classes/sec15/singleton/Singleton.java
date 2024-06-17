package ch06_.classes.sec15.singleton;
/*
 * 소프트웨어 디자인 패턴에서 싱글턴 패턴(Singleton pattern)을 따르는 클래스는,
 *  생성자가 여러 차례 호출되더라도 실제로 생성되는 객체는 하나이고 
 *  최초 생성 이후에 호출된 생성자는 최초의 생성자가 생성한 객체를 리턴한다. 
 *  이와 같은 디자인 유형을 싱글턴 패턴이라고 한다. 주로 공통된 객체를 여러개 
 *  생성해서 사용하는 DBCP(DataBase Connection Pool)와 같은 상황에서 많이 사용된다.
 */


public class Singleton { // 글톤 패턴은 객체 지향 프로그래밍에서 특정 클래스가
	//단 하나만의 인스턴스를 생성하여 사용하기 위한 패턴
	
	//private 접근 권한을 갖는 정적 필드 선언과 초기화
	private static Singleton singleton = new Singleton(); // 최초에 1개만 만들어놓음. 

	//private 접근 권한을 갖는 생성자 선언
	private Singleton() {
		System.out.println("Singleton() 생성자");  // 3번실행 됨. 
	}

	//public 접근 권한을 갖는 정적 메소드 선언
	public static Singleton getInstance() {
		return singleton;
	}
	
	
	public static void main(String[] args) {
		Singleton st1 = new Singleton(); 
 		Singleton st2 = new Singleton();
 		System.out.println(st1 == st2); // false
	} 
	
}