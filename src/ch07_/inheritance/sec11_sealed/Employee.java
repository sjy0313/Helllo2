package ch07_.inheritance.sec11_sealed;
// Permitted class Docker does not declare ch07_.inheritance.sec11_sealed.Person as direct super class

//public final class Employee extends Person {
public non-sealed class Employee extends Person {
	@Override
	public void work() {
		System.out.println("제품을 생산합니다.");
	}
}