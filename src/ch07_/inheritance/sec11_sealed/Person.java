
package ch07_.inheritance.sec11_sealed;
/*
 * Java 15
 * 봉인된 클래스: sealed class
 *    - 상속 제한
 *    - 자식 클래스(subclass)를 지정된 클래스로 한정
 *    - 무분별한 상속을 통제
 */

// 부모클래스(Person)를 상속하는 자식클래스를 Employee, Manager로 제한
public sealed class Person permits Employee, Manager {
	//필드
	public String name;

	//메소드
	
	public void work() {
		System.out.println("하는 일이 결정되지 않았습니다.");
	}
}

